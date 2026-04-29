"""
spades_engine.py
Core Spades game engine
Covers: deck, hands, bidding, trick-taking, scoring, state observation
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

class Suit(IntEnum):
    CLUBS    = 0
    DIAMONDS = 1
    HEARTS   = 2
    SPADES   = 3

class Rank(IntEnum):
    TWO   = 2;  THREE = 3;  FOUR  = 4;  FIVE  = 5
    SIX   = 6;  SEVEN = 7;  EIGHT = 8;  NINE  = 9
    TEN   = 10; JACK  = 11; QUEEN = 12; KING  = 13; ACE = 14

SUIT_SYMBOLS = {Suit.CLUBS: "♣", Suit.DIAMONDS: "♦", Suit.HEARTS: "♥", Suit.SPADES: "♠"}
RANK_SYMBOLS = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
}

@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def index(self) -> int:
        """Unique 0-51 integer for encoding."""
        return self.suit * 13 + (self.rank - 2)


FULL_DECK: list[Card] = [
    Card(rank, suit)
    for suit in Suit
    for rank in Rank
]

def new_shuffled_deck() -> list[Card]:
    deck = list(FULL_DECK)
    random.shuffle(deck)
    return deck


# ---------------------------------------------------------------------------
# Bidding
# ---------------------------------------------------------------------------

# Bid sentinel values
NIL       = 0   # predict zero tricks, score 100/-100
BLIND_NIL = -1  # nil without seeing cards, score 200/-200

def bid_is_nil(bid: int) -> bool:
    return bid in (NIL, BLIND_NIL)

def bid_label(bid: int) -> str:
    if bid == BLIND_NIL: return "Blind Nil"
    if bid == NIL:       return "Nil"
    return str(bid)


# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

NUM_PLAYERS   = 4
CARDS_PER_HAND = 13
TRICKS_PER_HAND = 13
BAG_LIMIT     = 10   # 10 bags = -100 penalty
WINNING_SCORE = 500  # first team to reach this wins (optional end condition)
TEAM_A        = (0, 2)   # players 0 and 2
TEAM_B        = (1, 3)   # players 1 and 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Omniscient game state. Agents receive a filtered observation.
    Players: 0=South, 1=West, 2=North, 3=East
    Teams:   0+2 vs 1+3
    """
    # Hands
    hands: list[list[Card]] = field(default_factory=lambda: [[] for _ in range(NUM_PLAYERS)])

    # Bidding
    bids: list[Optional[int]] = field(default_factory=lambda: [None] * NUM_PLAYERS)
    bidding_complete: bool = False

    # Trick tracking
    current_trick: list[Optional[Card]] = field(default_factory=lambda: [None] * NUM_PLAYERS)
    trick_leader: int = 0          # player who leads the current trick
    tricks_won: list[int] = field(default_factory=lambda: [0] * NUM_PLAYERS)

    # Spades broken
    spades_broken: bool = False

    # Turn management
    current_player: int = 0        # whose turn to act (bid or play)
    phase: str = "deal"            # "deal" | "bid" | "play" | "score"

    # Scores (cumulative across rounds)
    team_scores: list[int] = field(default_factory=lambda: [0, 0])  # team 0, team 1
    team_bags: list[int] = field(default_factory=lambda: [0, 0])

    # History (for learning)
    tricks_played: list[list[Card]] = field(default_factory=list)   # completed tricks
    trick_winners: list[int] = field(default_factory=list)

    # Round number
    round_number: int = 0

    def team_of(self, player: int) -> int:
        return 0 if player in TEAM_A else 1

    def partner_of(self, player: int) -> int:
        return (player + 2) % 4

    def played_cards(self) -> list[Card]:
        """All cards played so far this round (in completed tricks)."""
        cards = []
        for trick in self.tricks_played:
            cards.extend(trick)
        return cards

    def current_trick_cards(self) -> list[tuple[int, Card]]:
        """(player_index, card) pairs for current trick, in play order."""
        result = []
        for i in range(NUM_PLAYERS):
            p = (self.trick_leader + i) % NUM_PLAYERS
            if self.current_trick[p] is not None:
                result.append((p, self.current_trick[p]))
        return result

    def tricks_in_current_trick(self) -> int:
        return sum(1 for c in self.current_trick if c is not None)

    def led_suit(self) -> Optional[Suit]:
        """Suit led in the current trick, or None if trick not started."""
        if self.current_trick[self.trick_leader] is not None:
            return self.current_trick[self.trick_leader].suit
        return None

    def observation(self, player: int) -> "Observation":
        """Return a partial-information view for the given player."""
        return Observation(state=self, player=player)


@dataclass
class Observation:
    """
    What a single player can see. This is the input to any agent.
    """
    state: GameState
    player: int

    @property
    def hand(self) -> list[Card]:
        return self.state.hands[self.player]

    @property
    def bids(self) -> list[Optional[int]]:
        return self.state.bids[:]

    @property
    def tricks_won(self) -> list[int]:
        return self.state.tricks_won[:]

    @property
    def spades_broken(self) -> bool:
        return self.state.spades_broken

    @property
    def led_suit(self) -> Optional[Suit]:
        return self.state.led_suit()

    @property
    def current_trick(self) -> list[Optional[Card]]:
        return self.state.current_trick[:]

    @property
    def played_cards(self) -> list[Card]:
        """
        Cards that are definitely known: completed tricks + cards already
        played in the *current* (in-progress) trick.

        current_trick holds stale values between tricks (it is overwritten at
        the start of each trick but never explicitly cleared at the end).
        We only include current_trick cards when a trick is actually in
        progress, i.e. the number of non-None slots is less than NUM_PLAYERS
        OR the current_trick cards are not already in tricks_played.
        Safest: only add current_trick cards that are NOT already in tricks_played.
        """
        played = self.state.played_cards()   # all completed tricks
        played_set = set(played)
        for c in self.state.current_trick:
            if c is not None and c not in played_set:
                played.append(c)
                played_set.add(c)
        return played

    def encode(self) -> list[float]:
        """
        Numeric feature vector for neural network input.
        Dimensions: 52 (hand) + 52 (seen played) + 52 (current trick) +
                    4 (bids/13) + 4 (tricks won/13) + 1 (spades broken) = 165
        """
        def card_bits(cards: list[Card]) -> list[float]:
            vec = [0.0] * 52
            for c in cards:
                vec[c.index] = 1.0
            return vec

        hand_vec    = card_bits(self.hand)
        played_vec  = card_bits(self.played_cards)
        trick_vec   = card_bits([c for c in self.current_trick if c is not None])
        bid_vec     = [(b / 13.0 if b is not None and b >= 0 else 0.0) for b in self.bids]
        tricks_vec  = [t / 13.0 for t in self.tricks_won]
        broken_vec  = [1.0 if self.spades_broken else 0.0]

        return hand_vec + played_vec + trick_vec + bid_vec + tricks_vec + broken_vec


# ---------------------------------------------------------------------------
# Legal move generation
# ---------------------------------------------------------------------------

def legal_plays(hand: list[Card], state: GameState) -> list[Card]:
    """
    Returns the list of cards the current player may legally play.
    Rules:
    - Must follow the led suit if possible.
    - Spades may not lead a trick until broken (unless only spades remain).
    - Any card may be played if you cannot follow suit.
    """
    led = state.led_suit()

    if led is None:
        # Leading the trick
        non_spades = [c for c in hand if c.suit != Suit.SPADES]
        if non_spades and not state.spades_broken:
            return non_spades
        return list(hand)   # only spades, or spades broken
    else:
        # Following
        same_suit = [c for c in hand if c.suit == led]
        return same_suit if same_suit else list(hand)


def legal_bids(hand: list[Card], blind_nil_allowed: bool = False) -> list[int]:
    """
    Valid bids: 0 (nil) through 13, plus -1 (blind nil) if allowed.
    In standard play, blind nil is only available before seeing your cards.
    """
    bids = list(range(0, 14))  # 0..13
    if blind_nil_allowed:
        bids = [BLIND_NIL] + bids
    return bids


# ---------------------------------------------------------------------------
# Trick evaluation
# ---------------------------------------------------------------------------

def trick_winner(trick: list[tuple[int, Card]], led_suit: Suit) -> int:
    """
    Given (player, card) pairs for a complete trick and the led suit,
    return the player index who wins.
    """
    spades = [(p, c) for p, c in trick if c.suit == Suit.SPADES]
    if spades:
        winner_pair = max(spades, key=lambda x: x[1].rank)
    else:
        on_suit = [(p, c) for p, c in trick if c.suit == led_suit]
        winner_pair = max(on_suit, key=lambda x: x[1].rank)
    return winner_pair[0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    team_deltas: list[int]   # score change for [team_A, team_B]
    bag_deltas: list[int]    # bags earned this round
    details: str             # human-readable summary

def score_round(state: GameState) -> RoundResult:
    """
    Compute score deltas for a completed round.

    Team bid = sum of non-nil bids for the two partners.
    Bags = tricks_won_by_team - team_bid  (if > 0).
    Nil scoring is per-player.
    """
    deltas = [0, 0]
    bag_deltas = [0, 0]
    lines = []

    for team_idx, players in enumerate([TEAM_A, TEAM_B]):
        p0, p1 = players
        bid0, bid1 = state.bids[p0], state.bids[p1]
        won0, won1 = state.tricks_won[p0], state.tricks_won[p1]

        # Handle nil bids first (scored individually, not counted in team bid)
        for p, bid, won in [(p0, bid0, won0), (p1, bid1, won1)]:
            if bid == NIL:
                if won == 0:
                    deltas[team_idx] += 100
                    lines.append(f"  P{p} Nil success: +100")
                else:
                    deltas[team_idx] -= 100
                    lines.append(f"  P{p} Nil fail ({won} tricks): -100")
            elif bid == BLIND_NIL:
                if won == 0:
                    deltas[team_idx] += 200
                    lines.append(f"  P{p} Blind Nil success: +200")
                else:
                    deltas[team_idx] -= 200
                    lines.append(f"  P{p} Blind Nil fail ({won} tricks): -200")

        # Team bid and tricks (nil bids excluded from both sides)
        team_bid  = sum(b for b in (bid0, bid1) if b is not None and b > 0)
        nil_won   = sum(w for p, w in [(p0, won0), (p1, won1)]
                        if bid_is_nil(state.bids[p]))
        team_won  = won0 + won1 - nil_won   # tricks attributable to non-nil bidders

        if team_bid == 0:
            # Both players bid nil — no team component
            pass
        elif team_won >= team_bid:
            bags = team_won - team_bid
            deltas[team_idx] += team_bid * 10 + bags
            bag_deltas[team_idx] = bags
            lines.append(
                f"  Team {team_idx} made bid {team_bid} ({team_won} tricks, {bags} bags): "
                f"+{team_bid * 10 + bags}"
            )
            # Bag penalty
            new_bags = state.team_bags[team_idx] + bags
            if new_bags >= BAG_LIMIT:
                deltas[team_idx] -= 100
                bag_deltas[team_idx] -= BAG_LIMIT  # reset count after penalty
                lines.append(f"  Team {team_idx} bag penalty: -100")
        else:
            deltas[team_idx] -= team_bid * 10
            lines.append(
                f"  Team {team_idx} set (bid {team_bid}, got {team_won}): "
                f"-{team_bid * 10}"
            )

    return RoundResult(
        team_deltas=deltas,
        bag_deltas=bag_deltas,
        details="\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

class SpadesGame:
    """
    Orchestrates a full game. Agents are callables:
      bid_agent(obs: Observation) -> int
      play_agent(obs: Observation, legal: list[Card]) -> Card
    """

    def __init__(
        self,
        bid_agents=None,
        play_agents=None,
        dealer: int = 0,
        verbose: bool = False,
    ):
        self.state = GameState()
        self.verbose = verbose
        # Default to random agents
        self.bid_agents  = bid_agents  or [random_bid_agent]  * NUM_PLAYERS
        self.play_agents = play_agents or [random_play_agent] * NUM_PLAYERS
        self.dealer = dealer

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def deal(self):
        deck = new_shuffled_deck()
        for i in range(NUM_PLAYERS):
            self.state.hands[i] = sorted(
                deck[i * CARDS_PER_HAND:(i + 1) * CARDS_PER_HAND]
            )
        self.state.phase = "bid"
        self.state.current_player = (self.dealer + 1) % NUM_PLAYERS
        self._log(f"\n=== Round {self.state.round_number + 1} ===")

    def run_bidding(self):
        start = (self.dealer + 1) % NUM_PLAYERS
        for i in range(NUM_PLAYERS):
            p = (start + i) % NUM_PLAYERS
            obs = self.state.observation(p)
            bid = self.bid_agents[p](obs)
            assert bid in legal_bids(self.state.hands[p], blind_nil_allowed=(bid == BLIND_NIL))
            self.state.bids[p] = bid
            self._log(f"  P{p} bids {bid_label(bid)}")
        self.state.bidding_complete = True
        self.state.phase = "play"
        self.state.trick_leader = (self.dealer + 1) % NUM_PLAYERS
        self.state.current_player = self.state.trick_leader

    def run_trick(self):
        leader = self.state.trick_leader
        self.state.current_trick = [None] * NUM_PLAYERS
        played_pairs = []

        for i in range(NUM_PLAYERS):
            p = (leader + i) % NUM_PLAYERS
            obs  = self.state.observation(p)
            legal = legal_plays(self.state.hands[p], self.state)
            card  = self.play_agents[p](obs, legal)
            assert card in legal, f"P{p} played illegal card {card}"
            self.state.hands[p].remove(card)
            self.state.current_trick[p] = card
            played_pairs.append((p, card))
            if card.suit == Suit.SPADES:
                self.state.spades_broken = True

        led = self.state.current_trick[leader].suit
        winner = trick_winner(played_pairs, led)
        self.state.tricks_won[winner] += 1
        self.state.tricks_played.append([c for _, c in played_pairs])
        self.state.trick_winners.append(winner)
        self.state.trick_leader = winner
        self.state.current_player = winner

        trick_str = "  ".join(str(c) for _, c in sorted(played_pairs))
        self._log(f"  Trick: {trick_str}  → P{winner} wins")

    def run_round(self):
        self.deal()
        self.run_bidding()
        for _ in range(TRICKS_PER_HAND):
            self.run_trick()
        result = score_round(self.state)
        for t in range(2):
            self.state.team_scores[t] += result.team_deltas[t]
            new_bags = self.state.team_bags[t] + result.bag_deltas[t]
            self.state.team_bags[t] = new_bags % BAG_LIMIT if new_bags >= BAG_LIMIT else new_bags
        self._log(f"\nRound scoring:\n{result.details}")
        self._log(f"Scores: Team A={self.state.team_scores[0]}, Team B={self.state.team_scores[1]}")
        self.state.round_number += 1
        # Reset for next round
        self.state.tricks_won = [0] * NUM_PLAYERS
        self.state.bids = [None] * NUM_PLAYERS
        self.state.bidding_complete = False
        self.state.spades_broken = False
        self.state.tricks_played = []
        self.state.trick_winners = []
        self.dealer = (self.dealer + 1) % NUM_PLAYERS
        return result

    def play_to_score(self, target: int = WINNING_SCORE, max_rounds: int = 50) -> int:
        """Play until a team reaches target. Returns winning team index."""
        for _ in range(max_rounds):
            self.run_round()
            for t in range(2):
                if self.state.team_scores[t] >= target:
                    self._log(f"\nTeam {t} wins with {self.state.team_scores[t]} points!")
                    return t
        # No winner — return team with higher score
        return 0 if self.state.team_scores[0] >= self.state.team_scores[1] else 1


# ---------------------------------------------------------------------------
# Built-in agents
# ---------------------------------------------------------------------------

def random_bid_agent(obs: Observation) -> int:
    """Bid randomly between 1 and 4 (avoids nil for simplicity)."""
    return random.randint(1, 4)

def random_play_agent(obs: Observation, legal: list[Card]) -> Card:
    """Play a random legal card."""
    return random.choice(legal)

def greedy_play_agent(obs: Observation, legal: list[Card]) -> Card:
    """
    Simple greedy: try to win the trick if possible, otherwise dump lowest.
    Knows about spades trumping.
    """
    led = obs.led_suit
    current_trick = obs.current_trick

    if led is None:
        # Leading — play highest non-spade, or highest spade
        non_spades = sorted([c for c in legal if c.suit != Suit.SPADES], key=lambda c: c.rank)
        if non_spades:
            return non_spades[-1]   # lead highest non-spade
        return max(legal, key=lambda c: c.rank)

    # Find highest card currently winning the trick
    played = [(p, c) for p, c in enumerate(current_trick) if c is not None]
    if not played:
        return random.choice(legal)

    current_best = trick_winner(played, led)
    best_card = current_trick[current_best]

    # Can we beat it?
    winning_plays = []
    for c in legal:
        if c.suit == Suit.SPADES and best_card.suit != Suit.SPADES:
            winning_plays.append(c)   # spade beats any non-spade
        elif c.suit == best_card.suit and c.rank > best_card.rank:
            winning_plays.append(c)

    if winning_plays:
        return min(winning_plays, key=lambda c: c.rank)  # win cheaply
    return min(legal, key=lambda c: c.rank)               # dump lowest