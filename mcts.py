"""
mcts.py
Information-Set MCTS (IS-MCTS) via determinization for Spades

Algorithm:
  For each move decision:
    1. Sample N determinizations (complete worlds consistent with observation)
    2. Run standard UCT tree search on each world (perfect-information)
    3. Aggregate action values across worlds → pick best action

  This is the "determinization" approach (also called Perfect Information MCTS
  or PIMC). It's simpler than full IS-MCTS.

Key classes:
  MCTSNode       — UCT tree node (perfect-info, per determinization)
  ISMCTSAgent    — aggregates across determinizations, exposes bid + play API
"""

from __future__ import annotations
import math
import random
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from spades_engine import (
    Card, Suit, GameState, Observation, SpadesGame, NUM_PLAYERS, TRICKS_PER_HAND,
    TEAM_A, TEAM_B, legal_plays, legal_bids, trick_winner, score_round,
    random_bid_agent, random_play_agent, greedy_play_agent,
    NIL, bid_label,
)
from spades_cli import determinize

# ── UCT constant ─────────────────────────────────────────────────────────────
UCT_C = 1.4   # exploration weight; rewards now in [0,1] so this is well-calibrated


# ── Simulation helpers ────────────────────────────────────────────────────────

def _finish_current_trick(state: GameState, rollout_fn) -> int:
    """
    Complete the in-progress trick with rollout_fn for all players who
    haven't yet played. Returns the trick winner index.
    """
    leader = state.trick_leader
    pairs  = []

    for i in range(NUM_PLAYERS):
        p = (leader + i) % NUM_PLAYERS
        c = state.current_trick[p]
        if c is None:
            legal = legal_plays(state.hands[p], state)
            c = rollout_fn(state.observation(p), legal)
            state.hands[p].remove(c)
            state.current_trick[p] = c
            if c.suit == Suit.SPADES:
                state.spades_broken = True
        pairs.append((p, c))

    led    = state.current_trick[leader].suit
    winner = trick_winner(pairs, led)
    state.tricks_won[winner] += 1
    state.trick_leader = winner
    return winner


def _play_full_game(state: GameState, rollout_fn) -> list[float]:
    """
    Simulate from current state to end using rollout_fn.
    Returns per-player normalised reward in [0,1].

    Reward = team_tricks_won / 13  (simple; replace with score delta on Day 4+)
    """
    sim = deepcopy(state)

    # Finish current in-progress trick if one is underway
    cards_in_trick = sum(1 for c in sim.current_trick if c is not None)
    if 0 < cards_in_trick < NUM_PLAYERS:
        _finish_current_trick(sim, rollout_fn)
        sim.current_trick = [None] * NUM_PLAYERS

    # Play remaining full tricks
    tricks_remaining = min(len(sim.hands[p]) for p in range(NUM_PLAYERS))
    for _ in range(tricks_remaining):
        leader = sim.trick_leader
        sim.current_trick = [None] * NUM_PLAYERS
        pairs = []
        for i in range(NUM_PLAYERS):
            p = (leader + i) % NUM_PLAYERS
            legal = legal_plays(sim.hands[p], sim)
            c = rollout_fn(sim.observation(p), legal)
            sim.hands[p].remove(c)
            sim.current_trick[p] = c
            pairs.append((p, c))
            if c.suit == Suit.SPADES:
                sim.spades_broken = True
        led    = sim.current_trick[leader].suit
        winner = trick_winner(pairs, led)
        sim.tricks_won[winner] += 1
        sim.trick_leader = winner
        sim.current_trick = [None] * NUM_PLAYERS

    # Reward: tricks won by each team, normalised to [0, 1].
    # score_round clusters tightly around bid*10 ± small bag noise,
    team_tricks = [
        sum(sim.tricks_won[p] for p in TEAM_A),
        sum(sim.tricks_won[p] for p in TEAM_B),
    ]
    team_reward = [t / TRICKS_PER_HAND for t in team_tricks]  # in [0, 1]

    rewards = []
    for p in range(NUM_PLAYERS):
        t = sim.team_of(p)
        rewards.append(team_reward[t])
    return rewards


# ── UCT Node ──────────────────────────────────────────────────────────────────

class MCTSNode:
    """
    A node in the UCT search tree for ONE determinization (perfect-info game).

    Each node represents a game state.
    Children are indexed by the Card played (or bid int).
    """

    __slots__ = (
        "state", "player", "parent", "action",
        "children", "untried_actions",
        "visit_count", "value_sum",
    )

    def __init__(
        self,
        state: GameState,
        player: int,          # player whose turn it is
        parent: Optional[MCTSNode] = None,
        action=None,          # Card or bid int that led here
    ):
        self.state          = state
        self.player         = player
        self.parent         = parent
        self.action         = action
        self.children:      dict = {}          # action → MCTSNode
        self.untried_actions: list = []
        self.visit_count:   int = 0
        self.value_sum:     float = 0.0

        # Populate untried actions
        if state.phase == "play":
            self.untried_actions = list(
                legal_plays(state.hands[player], state)
            )
        elif state.phase == "bid":
            self.untried_actions = list(legal_bids(state.hands[player]))

    # ── UCT formula ───────────────────────────────────────────────────────────

    def uct_score(self, parent_visits: int, c: float = UCT_C) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploit = self.value_sum / self.visit_count
        explore = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploit + explore

    def best_child(self, c: float = UCT_C) -> "MCTSNode":
        return max(
            self.children.values(),
            key=lambda n: n.uct_score(self.visit_count, c),
        )

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return all(len(self.state.hands[p]) == 0 for p in range(NUM_PLAYERS))


# ── Tree operations ──────────────────────────────────────────────────────────

def _apply_action(state: GameState, player: int, action) -> tuple[GameState, int]:
    """
    Apply action (Card or bid int) to state.
    Returns (new_state, next_player).
    Modifies a deepcopy — does NOT mutate state.
    """
    s = deepcopy(state)

    if s.phase == "bid":
        s.bids[player] = action
        next_p = (player + 1) % NUM_PLAYERS
        # Check if bidding complete
        if all(b is not None for b in s.bids):
            s.bidding_complete = True
            s.phase = "play"
            # Leader is player left of dealer; we approximate as player 0
            # (exact dealer tracking isn't critical for MCTS quality)
            s.trick_leader = next_p % NUM_PLAYERS
            s.current_player = s.trick_leader
            next_p = s.trick_leader
        else:
            s.current_player = next_p
        return s, next_p

    # Play phase
    card = action
    assert card in s.hands[player], f"Card {card} not in hand of P{player}"
    s.hands[player].remove(card)
    s.current_trick[player] = card
    if card.suit == Suit.SPADES:
        s.spades_broken = True

    # Check if trick complete
    cards_played = sum(1 for c in s.current_trick if c is not None)
    if cards_played == NUM_PLAYERS:
        led    = s.current_trick[s.trick_leader].suit
        pairs  = [(p, s.current_trick[p]) for p in range(NUM_PLAYERS)]
        winner = trick_winner(pairs, led)
        s.tricks_won[winner] += 1
        s.tricks_played.append([c for _, c in sorted(pairs)])
        s.trick_winners.append(winner)
        s.trick_leader = winner
        s.current_player = winner
        s.current_trick = [None] * NUM_PLAYERS
        next_p = winner
    else:
        # Next player in trick rotation
        next_p = (player + 1) % NUM_PLAYERS
        while s.current_trick[next_p] is not None:
            next_p = (next_p + 1) % NUM_PLAYERS
        s.current_player = next_p

    return s, next_p


def _expand(node: MCTSNode) -> MCTSNode:
    """Pick one untried action, create a child node."""
    action = node.untried_actions.pop(
        random.randrange(len(node.untried_actions))
    )
    new_state, next_player = _apply_action(node.state, node.player, action)
    child = MCTSNode(
        state=new_state,
        player=next_player,
        parent=node,
        action=action,
    )
    node.children[action] = child
    return child


def _backprop(node: MCTSNode, rewards: list[float]):
    """Walk up the tree, updating visit counts and values."""
    n = node
    while n is not None:
        n.visit_count += 1
        n.value_sum   += rewards[n.player]
        n = n.parent


def run_mcts(
    root_state: GameState,
    root_player: int,
    n_simulations: int,
    rollout_fn=None,
    time_limit: float = None,   # seconds; overrides n_simulations if set
) -> dict:
    """
    Run UCT search from root_state for root_player.
    Returns a dict: action → (total_value, visit_count)
    """
    rollout_fn = rollout_fn or greedy_play_agent
    root = MCTSNode(root_state, root_player)

    deadline = (time.time() + time_limit) if time_limit else None
    sims_done = 0

    while True:
        if deadline and time.time() >= deadline:
            break
        if not deadline and sims_done >= n_simulations:
            break

        # 1. Selection
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = _expand(node)

        # 3. Simulation (rollout)
        rewards = _play_full_game(node.state, rollout_fn)

        # 4. Backpropagation
        _backprop(node, rewards)
        sims_done += 1

    # Collect action stats from root's children
    stats = {}
    for action, child in root.children.items():
        stats[action] = (child.value_sum, child.visit_count)
    return stats


# ── IS-MCTS Agent ─────────────────────────────────────────────────────────────

class ISMCTSAgent:
    """
    Information-Set MCTS agent for Spades.

    On each decision:
      - Sample `n_determinizations` complete worlds from the observation.
      - Run `n_simulations` UCT iterations on each world.
      - Aggregate visit counts across worlds → pick action with most visits.

    Parameters
    ----------
    n_determinizations : int
        Worlds to sample. More = better quality, slower. Start with 20.
    n_simulations : int
        UCT iterations per world. Start with 50.
    time_limit : float | None
        If set, run MCTS for this many seconds per world instead of fixed sims.
    rollout_fn : callable
        Policy used in rollout phase. greedy_play_agent by default.
    bid_strategy : str
        "mcts"    — run MCTS over bid actions (slow, thorough)
        "heuristic" — fast rule-based bid (recommended for deadline)
    """

    def __init__(
        self,
        n_determinizations: int = 20,
        n_simulations: int = 75,
        n_sims: int = None,
        time_limit: float = None,
        rollout_fn=None,
        bid_strategy: str = "heuristic",
        verbose: bool = False,
        seed: int = None,
    ):
        self.n_det   = n_determinizations
        self.n_sims  = n_sims if n_sims is not None else n_simulations

        self.t_limit = time_limit
        self.rollout = rollout_fn or greedy_play_agent
        self.bid_strat = bid_strategy
        self.verbose = verbose
        self.rng = random.Random(seed)

    # ── Bid ───────────────────────────────────────────────────────────────────

    def bid(self, obs: Observation) -> int:
        if self.bid_strat == "heuristic":
            return self._heuristic_bid(obs)
        # MCTS bid: aggregate over determinizations
        return self._mcts_bid(obs)

    def _heuristic_bid(self, obs: Observation) -> int:
        """
        Simple but effective hand-evaluation heuristic.
        Counts sure tricks + partial credit for high cards.
        """
        hand = obs.hand
        bid = 0

        # Group by suit
        suits: dict[Suit, list[Card]] = {s: [] for s in Suit}
        for c in hand:
            suits[c.suit].append(c)
        for s in Suit:
            suits[s].sort(key=lambda c: -c.rank)

        from spades_engine import Rank

        # Spades: each spade is roughly worth 0.5-1 trick
        sp = suits[Suit.SPADES]
        for i, c in enumerate(sp):
            if c.rank >= Rank.QUEEN:
                bid += 1
            elif c.rank >= Rank.NINE and i < 3:
                bid += 1
            elif len(sp) >= 5:
                bid += 0   # long spades covered by length tricks

        # Aces in non-spade suits
        for s in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS]:
            cards = suits[s]
            if not cards:
                continue
            if cards[0].rank == Rank.ACE:
                bid += 1
                cards = cards[1:]
            if cards and cards[0].rank == Rank.KING and len(cards) >= 2:
                bid += 1
                cards = cards[1:]
            # Void / singleton: potential ruff
            if len(suits[s]) == 0:
                bid += 1
            elif len(suits[s]) == 1 and suits[Suit.SPADES]:
                bid += 0.5

        bid = max(1, round(bid))   # never bid 0 from heuristic (nil needs care)
        return min(bid, 13)

    def _mcts_bid(self, obs: Observation) -> int:
        visit_totals: dict[int, int] = defaultdict(int)
        for _ in range(self.n_det):
            world = determinize(obs, self.rng)
            world.phase = "bid"
            world.bids  = list(obs.bids)  # fill in known bids
            stats = run_mcts(
                world, obs.player,
                n_simulations=self.n_sims,
                rollout_fn=self.rollout,
                time_limit=self.t_limit,
            )
            for action, (_, visits) in stats.items():
                visit_totals[action] += visits

        if not visit_totals:
            return self._heuristic_bid(obs)

        best = max(visit_totals, key=visit_totals.get)
        if self.verbose:
            top = sorted(visit_totals.items(), key=lambda x: -x[1])[:4]
            print(f"  [MCTS bid] top actions: {[(bid_label(a), v) for a,v in top]}")
        return best

    # ── Play ──────────────────────────────────────────────────────────────────

    def play(self, obs: Observation, legal: list[Card]) -> Card:
        if len(legal) == 1:
            return legal[0]

        visit_totals: dict[Card, int] = defaultdict(int)
        value_totals: dict[Card, float] = defaultdict(float)

        for _ in range(self.n_det):
            world = determinize(obs, self.rng)
            # Sync world state (bids, spades broken, trick state)
            world.bids          = list(obs.bids)
            world.spades_broken = obs.spades_broken
            world.current_trick = list(obs.current_trick)
            world.phase         = "play"
            world.current_player = obs.player

            stats = run_mcts(
                world, obs.player,
                n_simulations=self.n_sims,
                rollout_fn=self.rollout,
                time_limit=self.t_limit,
            )
            for action, (val, visits) in stats.items():
                if action in legal:
                    visit_totals[action] += visits
                    value_totals[action] += val

        if not visit_totals:
            return self.rollout(obs, legal)

        # Primary: most visits (robust child selection, AlphaZero)
        best = max(legal, key=lambda c: visit_totals.get(c, 0))

        if self.verbose:
            top = sorted(
                [(c, visit_totals[c], value_totals[c]) for c in legal],
                key=lambda x: -x[1],
            )[:4]
            print(f"  [MCTS play] top cards: {[(str(c), v, f'{val:.2f}') for c,v,val in top]}")

        return best

    # ── Agent interface (matches engine callback signatures) ──────────────────

    def as_bid_agent(self):
        """Return a bid_agent callable for SpadesGame."""
        def agent(obs: Observation) -> int:
            return self.bid(obs)
        return agent

    def as_play_agent(self):
        """Return a play_agent callable for SpadesGame."""
        def agent(obs: Observation, legal: list[Card]) -> Card:
            return self.play(obs, legal)
        return agent


# ── Quick benchmark ──────────────────────────────────────────────────────────

def benchmark(
    n_games: int = 10,
    n_det: int = 10,
    n_sims: int = 30,
    target: int = 200,
    verbose: bool = False,
):
    """
    Pit MCTS (Team A: players 0+2) against greedy (Team B: players 1+3).
    Prints win rate and average game length.
    """
    from collections import Counter
    import sys

    agent = ISMCTSAgent(
        n_determinizations=n_det,
        n_simulations=n_sims,
        verbose=verbose,
    )

    wins    = Counter()
    rounds  = []

    for i in range(n_games):
        g = SpadesGame(
            bid_agents=[
                agent.as_bid_agent(), random_bid_agent,
                agent.as_bid_agent(), random_bid_agent,
            ],
            play_agents=[
                agent.as_play_agent(), greedy_play_agent,
                agent.as_play_agent(), greedy_play_agent,
            ],
        )
        winner = g.play_to_score(target=target, max_rounds=20)
        wins[winner] += 1
        rounds.append(g.state.round_number)
        pct = (i + 1) / n_games * 100
        sys.stdout.write(
            f"\r  Game {i+1}/{n_games}  MCTS wins: {wins[0]}  "
            f"Greedy wins: {wins[1]}  ({pct:.0f}%)"
        )
        sys.stdout.flush()

    print()
    sep = "─" * 50
    print(sep)
    print(f"  MCTS  (Team A): {wins[0]}/{n_games}  ({wins[0]/n_games*100:.1f}%)")
    print(f"  Greedy(Team B): {wins[1]}/{n_games}  ({wins[1]/n_games*100:.1f}%)")
    print(f"  Avg rounds/game: {sum(rounds)/len(rounds):.1f}")
    print(sep)
    return wins


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IS-MCTS Spades agent")
    parser.add_argument("--games",  type=int, default=10,  help="Games to benchmark")
    parser.add_argument("--det",    type=int, default=10,  help="Determinizations per move")
    parser.add_argument("--sims",   type=int, default=30,  help="MCTS sims per determinization")
    parser.add_argument("--target", type=int, default=200, help="Score target")
    parser.add_argument("--verbose",action="store_true",   help="Print MCTS debug info")
    args = parser.parse_args()

    print(f"\nIS-MCTS benchmark: {args.games} games, "
          f"{args.det} det × {args.sims} sims\n")
    benchmark(
        n_games=args.games,
        n_det=args.det,
        n_sims=args.sims,
        target=args.target,
        verbose=args.verbose,
    )