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
from typing import Optional

from spades_engine import (
    Card, Suit, GameState, Observation, SpadesGame, NUM_PLAYERS, TRICKS_PER_HAND,
    TEAM_A, TEAM_B, legal_plays, legal_bids, trick_winner,
    random_bid_agent, greedy_play_agent,
    bid_label,
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
    pairs = []

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

    led = state.current_trick[leader].suit
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

    # Finish partial trick
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

        led = sim.current_trick[leader].suit
        winner = trick_winner(pairs, led)
        sim.tricks_won[winner] += 1
        sim.trick_leader = winner
        sim.current_trick = [None] * NUM_PLAYERS

    # Team scoring
    team_a_tricks = sum(sim.tricks_won[p] for p in TEAM_A)
    team_a_bid = sum(sim.bids[p] for p in TEAM_A if sim.bids[p] is not None)

    team_b_tricks = sum(sim.tricks_won[p] for p in TEAM_B)
    team_b_bid = sum(sim.bids[p] for p in TEAM_B if sim.bids[p] is not None)

    res_a = 1.0 if team_a_tricks >= team_a_bid else (team_a_tricks / 20.0)
    res_b = 1.0 if team_b_tricks >= team_b_bid else (team_b_tricks / 20.0)

    rewards = [0.0] * NUM_PLAYERS
    for p in range(NUM_PLAYERS):
        rewards[p] = res_a if p in TEAM_A else res_b

    return rewards


# ── MCTS Node ────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = (
        "state", "player", "parent", "action",
        "children", "untried_actions",
        "visit_count", "value_sum",
    )

    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

        if state.phase == "play":
            self.untried_actions = list(legal_plays(state.hands[player], state))
        else:
            self.untried_actions = list(legal_bids(state.hands[player]))

    def uct_score(self, parent_visits: int, c: float = UCT_C) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploit = self.value_sum / self.visit_count
        explore = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploit + explore

    def best_child(self):
        return max(self.children.values(),
                   key=lambda n: n.uct_score(self.visit_count))

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return all(len(self.state.hands[p]) == 0 for p in range(NUM_PLAYERS)) \
            and all(c is None for c in self.state.current_trick)


# ── Tree operations ──────────────────────────────────────────────────────────

def _apply_action(state, player, action):
    s = deepcopy(state)

    if s.phase == "bid":
        s.bids[player] = action
        next_p = (player + 1) % NUM_PLAYERS

        if all(b is not None for b in s.bids):
            s.phase = "play"
            s.trick_leader = next_p
            s.current_player = next_p
            return s, next_p

        s.current_player = next_p
        return s, next_p

    # play
    s.hands[player].remove(action)
    s.current_trick[player] = action

    if action.suit == Suit.SPADES:
        s.spades_broken = True

    if all(c is not None for c in s.current_trick):
        led = s.current_trick[s.trick_leader].suit
        pairs = [(p, s.current_trick[p]) for p in range(NUM_PLAYERS)]
        winner = trick_winner(pairs, led)

        s.tricks_won[winner] += 1
        s.trick_leader = winner
        s.current_player = winner
        s.current_trick = [None] * NUM_PLAYERS
        return s, winner

    next_p = (player + 1) % NUM_PLAYERS
    while s.current_trick[next_p] is not None:
        next_p = (next_p + 1) % NUM_PLAYERS

    s.current_player = next_p
    return s, next_p


def _expand(node):
    action = node.untried_actions.pop()
    new_state, next_player = _apply_action(node.state, node.player, action)
    child = MCTSNode(new_state, next_player, node, action)
    node.children[action] = child
    return child


def _backprop(node, rewards, root_player):
    while node is not None:
        node.visit_count += 1
        node.value_sum += rewards[root_player]
        node = node.parent


# ── MCTS ─────────────────────────────────────────────────────────────────────

def run_mcts(root_state, root_player, n_simulations, rng, rollout_fn=None):
    def rollout(obs, legal):
        if rng.random() < 0.7:
            return greedy_play_agent(obs, legal)
        return rng.choice(legal)

    rollout_fn = rollout_fn or rollout
    root = MCTSNode(root_state, root_player)

    for _ in range(n_simulations):
        node = root

        # selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # expansion
        if not node.is_terminal():
            node = _expand(node)

        rewards = _play_full_game(node.state, rollout_fn)
        _backprop(node, rewards, root_player)

    return {a: (c.value_sum, c.visit_count) for a, c in root.children.items()}


# ── Agent ────────────────────────────────────────────────────────────────────

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
    def __init__(self, n_det=20, n_sims=75, seed=None, verbose=False):
        self.n_det = n_det
        self.n_sims = n_sims
        self.rng = random.Random(seed)
        self.verbose = verbose

    def bid(self, obs):
        return self._heuristic_bid(obs)

    def _heuristic_bid(self, obs):
        from spades_engine import Rank

        suits = {s: [] for s in Suit}
        for c in obs.hand:
            suits[c.suit].append(c)

        for s in Suit:
            suits[s].sort(key=lambda c: -c.rank)

        bid = 0

        sp = suits[Suit.SPADES]
        for i, c in enumerate(sp):
            if c.rank >= Rank.QUEEN:
                bid += 1
            elif c.rank >= Rank.NINE and i < 3:
                bid += 1

        for s in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS]:
            cards = suits[s]

            if len(cards) == 0:
                bid += 1
                continue

            if cards[0].rank == Rank.ACE:
                bid += 1
                cards = cards[1:]

            if cards and cards[0].rank == Rank.KING and len(cards) >= 2:
                bid += 1

            if len(cards) == 1 and sp:
                bid += 0.5

        return max(1, min(13, round(bid)))

    def play(self, obs, legal):
        totals = defaultdict(int)

        for _ in range(self.n_det):
            world = determinize(obs, self.rng)
            world.bids = list(obs.bids)
            world.phase = "play"
            world.current_player = obs.player
            world.current_trick = list(obs.current_trick)
            world.spades_broken = obs.spades_broken

            stats = run_mcts(world, obs.player, self.n_sims, self.rng)

            for a, (_, v) in stats.items():
                if a in legal:
                    totals[a] += v

        return max(legal, key=lambda a: totals[a]) if totals else self.rng.choice(legal)

    def as_bid_agent(self):
        return lambda obs: self.bid(obs)

    def as_play_agent(self):
        return lambda obs, legal: self.play(obs, legal)
    
if __name__ == "__main__":
    import argparse
    from collections import Counter
    import sys

    parser = argparse.ArgumentParser(description="IS-MCTS Spades agent")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--det", type=int, default=10)
    parser.add_argument("--sims", type=int, default=30)
    args = parser.parse_args()

    agent = ISMCTSAgent(n_det=args.det, n_sims=args.sims)

    wins = Counter()

    print(f"\nRunning {args.games} games...\n")

    for i in range(args.games):
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

        winner = g.play_to_score(target=200)
        wins[winner] += 1

        sys.stdout.write(
            f"\rGame {i+1}/{args.games} | MCTS: {wins[0]} | Greedy: {wins[1]}"
        )
        sys.stdout.flush()

    print("\n\nResults:")
    print(f"MCTS wins: {wins[0]}")
    print(f"Greedy wins: {wins[1]}")