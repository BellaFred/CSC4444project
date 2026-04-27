"""
spades_cli.py
Human-playable CLI + determinization utilities

Usage:
  python spades_cli.py              # human vs 3 random agents
  python spades_cli.py --greedy     # human vs 3 greedy agents
  python spades_cli.py --watch      # watch 4 greedy agents play
  python spades_cli.py --sim 100    # simulate 100 random vs greedy games, print stats
"""

from __future__ import annotations
import argparse
import random
import sys
from copy import deepcopy
from typing import Optional

from spades_engine import (
    Card, Suit, Rank, GameState, Observation, SpadesGame,
    FULL_DECK, NUM_PLAYERS, TRICKS_PER_HAND, TEAM_A, TEAM_B,
    legal_plays, legal_bids, score_round,
    NIL, BLIND_NIL, bid_label,
    random_bid_agent, random_play_agent, greedy_play_agent,
)

# ── ANSI colours ────────────────────────────────────────────────────────────
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

SUIT_COLOR = {
    Suit.CLUBS:    "",
    Suit.DIAMONDS: RED,
    Suit.HEARTS:   RED,
    Suit.SPADES:   "",
}

def colored_card(card: Card) -> str:
    c = SUIT_COLOR[card.suit]
    return f"{c}{card}{RESET}"

def hand_display(hand: list[Card], show_index: bool = True) -> str:
    suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
    groups = []
    for s in suits:
        cards = sorted([c for c in hand if c.suit == s], key=lambda c: -c.rank)
        if cards:
            label = str(s.name.title()[0])  # S / H / D / C
            cards_str = " ".join(colored_card(c) for c in cards)
            groups.append(f"{DIM}{label}{RESET} {cards_str}")
    return "  ".join(groups)

def numbered_hand(hand: list[Card]) -> str:
    lines = []
    for i, c in enumerate(hand):
        lines.append(f"  {BOLD}{i+1:2}{RESET}. {colored_card(c)}")
    return "\n".join(lines)

PLAYER_NAMES = ["You (S)", "West", "North", "East"]
TEAM_LABELS  = ["Team A (You+North)", "Team B (West+East)"]

def separator(char="─", width=60):
    print(f"{DIM}{char * width}{RESET}")

# ── Human agents ────────────────────────────────────────────────────────────

def human_bid_agent(obs: Observation) -> int:
    hand = obs.hand
    print(f"\n{BOLD}Your hand:{RESET}")
    print(f"  {hand_display(hand)}\n")
    print(f"  Bids so far: " + ", ".join(
        f"P{i}={bid_label(b)}" for i, b in enumerate(obs.bids) if b is not None
    ) or "  (you bid first)")

    while True:
        raw = input(f"\n  Your bid (0=Nil, 1-13): ").strip()
        try:
            bid = int(raw)
            if bid in legal_bids(hand):
                return bid
            print(f"  {RED}Invalid bid. Enter 0-13.{RESET}")
        except ValueError:
            print(f"  {RED}Enter a number.{RESET}")

def human_play_agent(obs: Observation, legal: list[Card]) -> Card:
    hand = obs.hand
    led  = obs.led_suit

    print(f"\n  {BOLD}Your hand:{RESET}")
    print(numbered_hand(hand))

    if led:
        print(f"\n  Led suit: {CYAN}{led.name.title()}{RESET}")

    print(f"\n  {BOLD}Legal plays:{RESET}")
    for i, c in enumerate(legal):
        marker = f"{CYAN}*{RESET}" if c in legal else " "
        print(f"  {marker} {i+1}. {colored_card(c)}")

    while True:
        raw = input(f"\n  Play card (1-{len(legal)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(legal):
                return legal[idx]
            print(f"  {RED}Enter 1-{len(legal)}.{RESET}")
        except ValueError:
            print(f"  {RED}Enter a number.{RESET}")

# ── Display helpers ──────────────────────────────────────────────────────────

def print_trick_state(state: GameState, phase: str = "mid"):
    """Show the current trick as a 2×2 grid (N/S/E/W layout)."""
    ct = state.current_trick
    leader = state.trick_leader

    def fmt(p):
        c = ct[p]
        if c is None:
            return "   "
        marker = f"{YELLOW}►{RESET}" if p == leader else " "
        return f"{marker}{colored_card(c)}"

    north = fmt(2); south = fmt(0); west = fmt(1); east = fmt(3)
    w = 18
    print(f"\n{'':^{w}}{north:^{w}}")
    print(f"  {west:<{w}}{south:^{w}}{east:>{w}}")
    print()

def print_scoreboard(state: GameState):
    separator()
    print(f"  {BOLD}SCOREBOARD{RESET}")
    for t, players in enumerate([TEAM_A, TEAM_B]):
        p0, p1 = players
        bids = [bid_label(state.bids[p]) if state.bids[p] is not None else "?" for p in players]
        won  = [str(state.tricks_won[p]) for p in players]
        score = state.team_scores[t]
        bags  = state.team_bags[t]
        print(
            f"  {TEAM_LABELS[t]:30}  "
            f"Bids: {bids[0]}+{bids[1]}  "
            f"Won: {won[0]}+{won[1]}  "
            f"Score: {BOLD}{score:+4}{RESET}  "
            f"Bags: {bags}"
        )
    separator()

# ── Game orchestrator (verbose, human-aware) ─────────────────────────────────

class VerboseSpadesGame(SpadesGame):
    """
    Extends SpadesGame with rich console output.
    Overrides deal / bidding / trick methods to print state.
    """

    def deal(self):
        super().deal()
        separator("═")
        print(f"  {BOLD}ROUND {self.state.round_number + 1}{RESET}  |  Dealer: {PLAYER_NAMES[self.dealer]}")
        separator("═")

    def run_bidding(self):
        print(f"\n{BOLD}── BIDDING ──{RESET}")
        start = (self.dealer + 1) % NUM_PLAYERS
        for i in range(NUM_PLAYERS):
            p = (start + i) % NUM_PLAYERS
            obs = self.state.observation(p)
            bid = self.bid_agents[p](obs)
            self.state.bids[p] = bid
            print(f"  {PLAYER_NAMES[p]:10} bids  {BOLD}{bid_label(bid)}{RESET}")
        self.state.bidding_complete = True
        self.state.phase = "play"
        self.state.trick_leader = (self.dealer + 1) % NUM_PLAYERS
        self.state.current_player = self.state.trick_leader

    def run_trick(self):
        leader = self.state.trick_leader
        self.state.current_trick = [None] * NUM_PLAYERS
        played_pairs = []

        trick_num = len(self.state.tricks_played) + 1
        print(f"\n{DIM}── Trick {trick_num} ──{RESET}")

        for i in range(NUM_PLAYERS):
            p = (leader + i) % NUM_PLAYERS
            obs   = self.state.observation(p)
            legal = legal_plays(self.state.hands[p], self.state)
            card  = self.play_agents[p](obs, legal)
            assert card in legal
            self.state.hands[p].remove(card)
            self.state.current_trick[p] = card
            played_pairs.append((p, card))
            if card.suit == Suit.SPADES:
                self.state.spades_broken = True
            if p != 0:  # don't echo human's own play
                print(f"  {PLAYER_NAMES[p]:10} plays {colored_card(card)}")

        led    = self.state.current_trick[leader].suit
        from spades_engine import trick_winner
        winner = trick_winner(played_pairs, led)
        self.state.tricks_won[winner] += 1
        self.state.tricks_played.append([c for _, c in played_pairs])
        self.state.trick_winners.append(winner)
        self.state.trick_leader = winner
        self.state.current_player = winner

        print(f"  {YELLOW}→ {PLAYER_NAMES[winner]} wins trick {trick_num}{RESET}")
        if self.state.spades_broken and not any(
            c.suit == Suit.SPADES for trick in self.state.tricks_played[:-1]
            for c in trick
        ):
            print(f"  {DIM}(Spades broken){RESET}")

    def run_round(self):
        self.deal()
        self.run_bidding()
        for _ in range(TRICKS_PER_HAND):
            self.run_trick()
        result = score_round(self.state)
        for t in range(2):
            self.state.team_scores[t] += result.team_deltas[t]
            new_bags = self.state.team_bags[t] + result.bag_deltas[t]
            self.state.team_bags[t] = new_bags % 10 if new_bags >= 10 else new_bags
        print(f"\n{BOLD}── Round result ──{RESET}")
        print(result.details)
        print_scoreboard(self.state)
        self.state.round_number += 1
        # Reset per-round state
        self.state.tricks_won      = [0] * NUM_PLAYERS
        self.state.bids            = [None] * NUM_PLAYERS
        self.state.bidding_complete = False
        self.state.spades_broken   = False
        self.state.tricks_played   = []
        self.state.trick_winners   = []
        self.dealer = (self.dealer + 1) % NUM_PLAYERS
        return result

    def play_to_score(self, target: int = 500, max_rounds: int = 50) -> int:
        for _ in range(max_rounds):
            self.run_round()
            for t in range(2):
                if self.state.team_scores[t] >= target:
                    separator("═")
                    print(f"\n  {BOLD}{YELLOW}{TEAM_LABELS[t]} wins!{RESET}  "
                          f"({self.state.team_scores[t]} points)\n")
                    separator("═")
                    return t
            input(f"\n  {DIM}[Press Enter for next round]{RESET}")
        return 0 if self.state.team_scores[0] >= self.state.team_scores[1] else 1


# ── Determinization (used by Day 3 MCTS) ────────────────────────────────────

def determinize(obs: Observation, rng: random.Random = None) -> GameState:
    """
    Given a player's observation, sample one *consistent* complete game state
    by assigning plausible hidden cards to the other three players.

    This is the key primitive for Information-Set MCTS (IS-MCTS):
    run many determinizations, solve each as a perfect-info game, aggregate.

    The sample respects:
      - Cards already in the observer's hand stay there.
      - Cards already played (visible) are not re-dealt.
      - Each hidden hand gets the same count as the original.
    """
    rng = rng or random.Random()
    state = obs.state

    # Cards known to be gone (played + current trick + observer's own hand)
    known_gone = set(obs.played_cards) | set(obs.hand)

    # Remaining hidden cards
    hidden = [c for c in FULL_DECK if c not in known_gone]
    rng.shuffle(hidden)

    # How many cards does each opponent currently hold?
    hand_sizes = [len(state.hands[p]) for p in range(NUM_PLAYERS)]
    hand_sizes[obs.player] = 0   # we'll skip the observer

    # Distribute hidden cards to opponents maintaining hand sizes
    new_state = deepcopy(state)
    cursor = 0
    for p in range(NUM_PLAYERS):
        if p == obs.player:
            continue
        size = hand_sizes[p]
        new_state.hands[p] = hidden[cursor:cursor + size]
        cursor += size

    return new_state


def multi_determinize(obs: Observation, n: int, rng: random.Random = None) -> list[GameState]:
    """Return n independent determinized states for IS-MCTS rollouts."""
    rng = rng or random.Random()
    return [determinize(obs, rng) for _ in range(n)]


def evaluate_action_by_rollout(
    obs: Observation,
    card: Card,
    n_worlds: int = 20,
    rollout_agent=None,
    rng: random.Random = None,
) -> float:
    """
    Quick Monte Carlo estimate of how good playing `card` is.
    Used as a sanity-check baseline before full MCTS is implemented.

    Returns average trick-win fraction for the observer's team
    across n_worlds determinizations.
    """
    rng = rng or random.Random()
    rollout_agent = rollout_agent or random_play_agent
    player = obs.player

    scores = []
    for world_state in multi_determinize(obs, n_worlds, rng):
        # Play out the rest of the game from this world
        sim = GameState()
        sim.__dict__.update(world_state.__dict__)

        # Force play of the chosen card for the current player
        legal = legal_plays(sim.hands[player], sim)
        if card not in legal:
            continue   # this world is inconsistent with the card — skip

        sim.hands[player].remove(card)
        sim.current_trick[player] = card
        if card.suit == Suit.SPADES:
            sim.spades_broken = True

        # Fill in the rest of the current trick with rollout agent
        leader = sim.trick_leader
        for i in range(NUM_PLAYERS):
            p = (leader + i) % NUM_PLAYERS
            if sim.current_trick[p] is not None or p == player:
                continue
            lc = legal_plays(sim.hands[p], sim)
            played = rollout_agent(sim.observation(p), lc)
            sim.hands[p].remove(played)
            sim.current_trick[p] = played
            if played.suit == Suit.SPADES:
                sim.spades_broken = True

        # Complete remaining tricks
        from spades_engine import trick_winner
        led = sim.current_trick[leader].suit
        trick_pairs = [(p, sim.current_trick[p]) for p in range(NUM_PLAYERS)]
        winner = trick_winner(trick_pairs, led)
        sim.tricks_won[winner] += 1
        sim.trick_leader = winner

        tricks_left = sum(len(sim.hands[p]) for p in range(NUM_PLAYERS)) // NUM_PLAYERS
        for _ in range(tricks_left):
            leader2 = sim.trick_leader
            sim.current_trick = [None] * NUM_PLAYERS
            pairs2 = []
            for i in range(NUM_PLAYERS):
                p = (leader2 + i) % NUM_PLAYERS
                lc = legal_plays(sim.hands[p], sim)
                played = rollout_agent(sim.observation(p), lc)
                sim.hands[p].remove(played)
                sim.current_trick[p] = played
                pairs2.append((p, played))
                if played.suit == Suit.SPADES:
                    sim.spades_broken = True
            led2 = sim.current_trick[leader2].suit
            winner2 = trick_winner(pairs2, led2)
            sim.tricks_won[winner2] += 1
            sim.trick_leader = winner2

        team = sim.team_of(player)
        team_players = TEAM_A if team == 0 else TEAM_B
        team_tricks = sum(sim.tricks_won[p] for p in team_players)
        scores.append(team_tricks / TRICKS_PER_HAND)

    return sum(scores) / len(scores) if scores else 0.0


# ── Simulation / stats ───────────────────────────────────────────────────────

def simulate_games(n: int, verbose: bool = False):
    """Simulate n games of greedy vs random, print win-rate stats."""
    from collections import Counter
    wins = Counter()
    total_rounds = []

    for game_i in range(n):
        g = SpadesGame(
            play_agents=[greedy_play_agent, random_play_agent,
                         greedy_play_agent, random_play_agent],
            verbose=verbose,
        )
        winner = g.play_to_score(target=500, max_rounds=30)
        wins[winner] += 1
        total_rounds.append(g.state.round_number)
        if (game_i + 1) % max(1, n // 10) == 0:
            pct = (game_i + 1) / n * 100
            sys.stdout.write(f"\r  Simulating... {pct:.0f}%")
            sys.stdout.flush()

    print(f"\r  Done.{' ' * 20}")
    separator()
    print(f"  {BOLD}Results ({n} games){RESET}")
    print(f"  {TEAM_LABELS[0]} (greedy): {wins[0]} wins  ({wins[0]/n*100:.1f}%)")
    print(f"  {TEAM_LABELS[1]} (random): {wins[1]} wins  ({wins[1]/n*100:.1f}%)")
    print(f"  Avg rounds per game: {sum(total_rounds)/len(total_rounds):.1f}")
    separator()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spades CLI")
    parser.add_argument("--greedy", action="store_true", help="Opponents use greedy agent")
    parser.add_argument("--watch",  action="store_true", help="Watch 4 greedy agents (no human)")
    parser.add_argument("--sim",    type=int, metavar="N", help="Simulate N games, print stats")
    parser.add_argument("--target", type=int, default=500, help="Winning score (default 500)")
    args = parser.parse_args()

    if args.sim:
        simulate_games(args.sim)
        return

    opponent = greedy_play_agent if args.greedy else random_play_agent
    opp_label = "greedy" if args.greedy else "random"

    if args.watch:
        bid_agents  = [random_bid_agent] * 4
        play_agents = [greedy_play_agent] * 4
        print(f"\n{BOLD}Watching 4 greedy agents...{RESET}  (target {args.target})")
    else:
        bid_agents  = [human_bid_agent,  random_bid_agent, random_bid_agent, random_bid_agent]
        play_agents = [human_play_agent, opponent, opponent, opponent]
        print(f"\n{BOLD}Spades  —  You vs 3 {opp_label} agents{RESET}")
        print(f"  Teams: {TEAM_LABELS[0]} vs {TEAM_LABELS[1]}")
        print(f"  First to {args.target} wins\n")

    game = VerboseSpadesGame(
        bid_agents=bid_agents,
        play_agents=play_agents,
        dealer=random.randint(0, 3),
    )
    game.play_to_score(target=args.target)


if __name__ == "__main__":
    main()