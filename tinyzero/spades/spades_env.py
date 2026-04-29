"""
spades_env.py  —  tinyzero-compatible Spades environment
==========================================================

Tested against s-casci/tinyzero (github.com/s-casci/tinyzero).

Key constraint from tinyzero's mcts.py:
    children_priors = policy_fn(game)[children_actions]
  Action encoding  (action_space = 66 total):
    0 – 13   bid value (0 = Nil, 1-13 = numeric bid)
    14 – 65  play card  (card.index + 14)
"""

from __future__ import annotations
import random
from copy import deepcopy
from typing import Optional
import numpy as np

from spades_engine import (
    Card, Suit, GameState,
    NUM_PLAYERS, TRICKS_PER_HAND, TEAM_A, TEAM_B,
    legal_plays, legal_bids, trick_winner,
    new_shuffled_deck, CARDS_PER_HAND, FULL_DECK,
    NIL,
)

# ── Action codec ──────────────────────────────────────────────────────────────

BID_OFFSET   = 0
CARD_OFFSET  = 14
ACTION_SPACE = 14 + 52   # 66

def card_to_action(card: Card) -> int:  return card.index + CARD_OFFSET
def action_to_card(action: int) -> Card: return FULL_DECK[action - CARD_OFFSET]
def bid_to_action(bid: int) -> int:     return bid + BID_OFFSET
def action_to_bid(action: int) -> int:  return action - BID_OFFSET
def is_bid_action(a: int) -> bool:      return BID_OFFSET <= a < CARD_OFFSET
def is_card_action(a: int) -> bool:     return CARD_OFFSET <= a < ACTION_SPACE


# ── Environment ───────────────────────────────────────────────────────────────

class SpadesEnv:
    """Single-round Spades environment compatible with tinyzero."""

    # Class-level attributes read by tinyzero / LinearNetwork
    observation_shape = (165,)
    action_space      = ACTION_SPACE   # LinearNetwork(obs_shape, game.action_space)
    n_players         = NUM_PLAYERS

    def __init__(self, dealer: int = 0, seed: int = None):
        self._dealer_start = dealer
        self._seed = seed
        # _state initialised before reset() so to_observation() is always safe
        self._state: GameState = GameState()
        self._history: list[GameState] = []
        self.reset()

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> "SpadesEnv":
        self._history.clear()
        self._state = GameState()

        if self._seed is not None:
            rng = random.Random(self._seed)
            deck = list(FULL_DECK)
            rng.shuffle(deck)
        else:
            deck = new_shuffled_deck()

        for i in range(NUM_PLAYERS):
            self._state.hands[i] = sorted(deck[i * CARDS_PER_HAND:(i + 1) * CARDS_PER_HAND])

        self._state.phase = "bid"
        self._state.current_player = (self._dealer_start + 1) % NUM_PLAYERS
        self._state.trick_leader   = (self._dealer_start + 1) % NUM_PLAYERS
        return self

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> None:
        self._history.append(deepcopy(self._state))
        s = self._state
        p = s.current_player

        if s.phase == "bid":
            if not is_bid_action(action):
                raise ValueError(f"Expected bid action 0-13, got {action}")
            self._apply_bid(p, action_to_bid(action))
        elif s.phase == "play":
            if not is_card_action(action):
                raise ValueError(f"Expected card action 14-65, got {action}")
            self._apply_play(p, action_to_card(action))
        else:
            raise RuntimeError(f"step() called in terminal phase '{s.phase}'")

    def _apply_bid(self, player: int, bid: int) -> None:
        s = self._state
        if bid not in legal_bids(s.hands[player]):
            raise ValueError(f"Illegal bid {bid} for player {player}")
        s.bids[player] = bid
        next_p = (player + 1) % NUM_PLAYERS
        if all(b is not None for b in s.bids):
            s.bidding_complete = True
            s.phase = "play"
            s.current_player = s.trick_leader
        else:
            s.current_player = next_p

    def _apply_play(self, player: int, card: Card) -> None:
        s = self._state
        if card not in legal_plays(s.hands[player], s):
            raise ValueError(f"Illegal play {card} for player {player}")
        s.hands[player].remove(card)
        s.current_trick[player] = card
        if card.suit == Suit.SPADES:
            s.spades_broken = True

        cards_in_trick = sum(1 for c in s.current_trick if c is not None)
        if cards_in_trick == NUM_PLAYERS:
            led    = s.current_trick[s.trick_leader].suit
            pairs  = [(p, s.current_trick[p]) for p in range(NUM_PLAYERS)]
            winner = trick_winner(pairs, led)
            s.tricks_won[winner] += 1
            s.tricks_played.append([c for _, c in sorted(pairs)])
            s.trick_winners.append(winner)
            s.trick_leader   = winner
            s.current_player = winner
            s.current_trick  = [None] * NUM_PLAYERS
            if all(len(s.hands[p]) == 0 for p in range(NUM_PLAYERS)):
                s.phase = "score"
        else:
            next_p = (player + 1) % NUM_PLAYERS
            while s.current_trick[next_p] is not None:
                next_p = (next_p + 1) % NUM_PLAYERS
            s.current_player = next_p

    # ── required interface ────────────────────────────────────────────────────

    def get_legal_actions(self) -> list[int]:
        """
        Returns legal actions as plain ints in [0, action_space).
        tinyzero indexes the policy array with these:
            policy_output[get_legal_actions()]
        """
        s = self._state
        p = s.current_player
        if s.phase == "bid":
            return [bid_to_action(b) for b in legal_bids(s.hands[p])]
        elif s.phase == "play":
            return [card_to_action(c) for c in legal_plays(s.hands[p], s)]
        return []

    def undo_last_action(self) -> None:
        if not self._history:
            raise IndexError("No actions to undo.")
        self._state = self._history.pop()

    def to_observation(self) -> np.ndarray:
        """165-dim float32 array for the current player."""
        obs = self._state.observation(self._state.current_player)
        return np.array(obs.encode(), dtype=np.float32)

    def get_result(self) -> Optional[float]:
        if self._state.phase != "score":
            return None
        ta = sum(self._state.tricks_won[p] for p in TEAM_A)
        tb = sum(self._state.tricks_won[p] for p in TEAM_B)
        if ta > tb: return  1.0
        if tb > ta: return -1.0
        return 0.0

    def get_first_person_result(self) -> Optional[float]:
        result = self.get_result()
        if result is None:
            return None
        team = self._state.team_of(self._state.current_player)
        return result if team == 0 else self.swap_result(result)

    @staticmethod
    def swap_result(result: Optional[float]) -> Optional[float]:
        if result is None or result == 0.0:
            return result
        return -result

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def current_player(self) -> int:       return self._state.current_player
    @property
    def current_player_index(self) -> int: return self._state.current_player
    @property
    def phase(self) -> str:                return self._state.phase
    @property
    def is_terminal(self) -> bool:         return self._state.phase == "score"
    @property
    def state(self) -> GameState:          return self._state

    def __repr__(self) -> str:
        s = self._state
        return f"SpadesEnv(phase={s.phase}, player={s.current_player}, tricks={s.tricks_won})"