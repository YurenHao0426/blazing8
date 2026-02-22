"""
Blazing Eights — multi-agent card game environment.

Cards:
  52 standard cards (4 suits × 13 ranks: A,2..10,J,Q,K)
  + 4 Swap cards (no suit, index 52-55)
  Total: 56 cards

Special cards:
  8  → Wild: player chooses a suit, next player must match that suit
  K  → All OTHER players draw 1 card from the deck
  Q  → Reverse direction (no effect in 2-player games)
  J  → Skip next player's turn
  Swap → Swap entire hand with next player (playable anytime; next card must match the card before the Swap)

Rules:
  - Match top card by suit OR rank (unless playing 8 or Swap)
  - Player may freely choose to draw even if they have playable cards
  - After drawing, player may play any playable card OR pass (end turn)
  - Each turn allows at most one draw
  - If no playable cards and deck is empty, player must pass
  - First player to empty hand wins
  - Initial hand: 5 cards each
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Card encoding
# ---------------------------------------------------------------------------
# Standard cards: index 0-51
#   suit = index // 13   (0=♠, 1=♥, 2=♦, 3=♣)
#   rank = index % 13    (0=A, 1=2, 2=3, ..., 9=10, 10=J, 11=Q, 12=K)
# Swap cards: index 52, 53, 54, 55

NUM_STANDARD = 52
NUM_SWAP = 4
NUM_CARDS = NUM_STANDARD + NUM_SWAP

RANK_A, RANK_J, RANK_Q, RANK_K = 0, 10, 11, 12
RANK_8 = 7  # rank index for 8 (0=A,1=2,...,7=8)

def card_suit(c: int) -> int:
    """Return suit of a standard card (0-3). Swap cards return -1."""
    return c // 13 if c < NUM_STANDARD else -1

def card_rank(c: int) -> int:
    """Return rank of a standard card (0-12). Swap cards return -1."""
    return c % 13 if c < NUM_STANDARD else -1

def is_swap(c: int) -> bool:
    return c >= NUM_STANDARD

def card_name(c: int) -> str:
    if is_swap(c):
        return f"SWAP-{c - NUM_STANDARD}"
    suits = "♠♥♦♣"
    ranks = ["A"] + [str(i) for i in range(2, 11)] + ["J", "Q", "K"]
    return f"{ranks[card_rank(c)]}{suits[card_suit(c)]}"


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
# Actions 0-55: play card with that index
# Actions 56-59: choose suit after playing an 8 (♠,♥,♦,♣)
# Action 60: draw a card
NUM_PLAY_ACTIONS = NUM_CARDS       # 0..55
NUM_SUIT_ACTIONS = 4               # 56..59
DRAW_ACTION = 60
PASS_ACTION = 61                   # skip turn (when deck empty & no playable card)
TOTAL_ACTIONS = 62


class BlazingEightsEnv:
    """
    Multi-agent environment for Blazing Eights.

    Designed for self-play RL. Call step() with the current player's action.
    The env tracks whose turn it is.
    """

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        assert 2 <= num_players <= 5
        self.num_players = num_players
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Build & shuffle deck
        # In 2-player games, remove Q cards (reverse has no effect)
        deck = [c for c in range(NUM_CARDS)
                if not (self.num_players == 2 and card_rank(c) == RANK_Q)]
        self.rng.shuffle(deck)

        # Deal 5 cards each
        self.hands: list[list[int]] = []
        idx = 0
        for _ in range(self.num_players):
            self.hands.append(sorted(deck[idx:idx + 5]))
            idx += 5

        # Find a non-special starting card for the discard pile
        # (avoid starting with 8, J, Q, K, or Swap)
        self.discard: list[int] = []
        start_card = None
        remaining = deck[idx:]
        for i, c in enumerate(remaining):
            if not is_swap(c) and card_rank(c) not in (RANK_8, RANK_J, RANK_Q, RANK_K):
                start_card = c
                remaining.pop(i)
                break
        if start_card is None:
            # Extremely unlikely; just use first card
            start_card = remaining.pop(0)
        self.discard.append(start_card)

        self.deck: list[int] = remaining
        self.current_player = int(self.rng.integers(0, self.num_players))
        self.direction = 1  # 1=clockwise, -1=counter-clockwise
        self.done = False
        self.winner = -1

        # State for wild-8: the chosen suit (None if top card is not a wild)
        self.active_suit: Optional[int] = None

        # Phase: "play" or "choose_suit"
        self.phase = "play"
        # Temp storage for the card that triggered choose_suit
        self._pending_8_player: Optional[int] = None

        # For K resolution
        self._pending_k = False

        # Track consecutive passes for stalemate detection
        self.consecutive_passes = 0

        # Track whether current player has already drawn this turn
        self.has_drawn_this_turn = False

        # Action history: records recent events visible to all players
        # Each entry: (player, event_type)
        #   event_type: 0=played_card, 1=(unused), 2=drew_card, 3=passed
        self.action_history: list[tuple[int, int]] = []
        self.max_history = 20  # keep last 20 events

        # Track swap visibility: after a swap, the swapper sees the received cards
        # This is informational; the obs encodes it
        self.swap_known_cards: dict[int, list[int]] = {}  # player -> known opponent cards

        return self._get_obs(self.current_player)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self, player: int) -> np.ndarray:
        """
        Observation vector for `player`:
          [0:56]    one-hot of cards in hand
          [56:60]   top card suit one-hot (or active_suit if wild)
          [60:73]   top card rank one-hot
          [73]      direction (0=cw, 1=ccw)
          [74:74+N-1] other players' hand sizes (normalized /20)
          [74+N-1]  deck size (normalized /56)
          [75+N-1]  phase: 0=play, 1=choose_suit
          [76+N-1 : 132+N-1] known cards of next player (from swap), 56 one-hot
          [132+N-1 : 132+N-1+(N-1)*5] per other player draw info:
              4 floats: last event one-hot (played/drew_played/drew_skipped/passed)
              1 float: consecutive draw-and-skip streak (/10)
        Padded to fixed 180.
        """
        obs = np.zeros(180, dtype=np.float32)

        # Hand
        for c in self.hands[player]:
            obs[c] = 1.0

        # Top card info (SWAP inherits previous card)
        top = self.discard[-1]
        eff = self._effective_top() if is_swap(top) else top
        if self.active_suit is not None:
            suit = self.active_suit
        elif not is_swap(eff):
            suit = card_suit(eff)
        else:
            suit = 0
        obs[56 + suit] = 1.0

        if not is_swap(eff) and self.active_suit is None:
            obs[60 + card_rank(eff)] = 1.0

        # Direction
        obs[73] = 0.0 if self.direction == 1 else 1.0

        # Other players' hand sizes
        for i in range(1, self.num_players):
            other = (player + i) % self.num_players
            obs[74 + i - 1] = len(self.hands[other]) / 20.0

        # Deck size
        obs[74 + self.num_players - 1] = len(self.deck) / 56.0

        # Phase
        obs[75 + self.num_players - 1] = 1.0 if self.phase == "choose_suit" else 0.0

        # Known cards of next player (from swap)
        offset = 76 + self.num_players - 1
        if player in self.swap_known_cards:
            for c in self.swap_known_cards[player]:
                obs[offset + c] = 1.0

        # Per other player: last event type + consecutive draw-skip streak
        # This encodes the "timer tells" — draw then skip means drawn card unplayable
        draw_info_offset = 132 + self.num_players - 1
        for i in range(1, self.num_players):
            other = (player + i) % self.num_players
            base = draw_info_offset + (i - 1) * 5

            # Scan history backwards for this player's events
            last_event = None
            consec_draw_skip = 0
            for p, evt in reversed(self.action_history):
                if p == other:
                    if last_event is None:
                        last_event = evt
                    if evt == 2:  # drew_and_skipped
                        consec_draw_skip += 1
                    else:
                        break

            if last_event is not None:
                obs[base + last_event] = 1.0
            obs[base + 4] = consec_draw_skip / 10.0

        return obs

    @staticmethod
    def obs_size() -> int:
        return 180

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def legal_actions(self, player: Optional[int] = None) -> list[int]:
        if self.done:
            return []
        if player is None:
            player = self.current_player

        if self.phase == "choose_suit":
            if player == self._pending_8_player:
                return [56, 57, 58, 59]
            else:
                return []

        actions = []
        hand = self.hands[player]
        top = self.discard[-1]

        for c in hand:
            if self._can_play(c, top):
                actions.append(c)

        if self.has_drawn_this_turn:
            # Already drew this turn: can play a card or pass (end turn)
            actions.append(PASS_ACTION)
        else:
            # Can always choose to draw instead of playing
            if self.deck or len(self.discard) > 1:
                actions.append(DRAW_ACTION)
            if not actions:
                # No playable cards and no deck: must pass
                actions.append(PASS_ACTION)
        return actions

    def _effective_top(self) -> int:
        """Find the last non-SWAP card in discard for matching purposes."""
        for c in reversed(self.discard):
            if not is_swap(c):
                return c
        return self.discard[-1]  # fallback (all swaps, shouldn't happen)

    def _can_play(self, card: int, top: int) -> bool:
        # Swap cards: always playable
        if is_swap(card):
            return True
        # 8s: always playable (wild)
        if card_rank(card) == RANK_8:
            return True
        # If active_suit is set (after a wild 8), must match that suit
        if self.active_suit is not None:
            return card_suit(card) == self.active_suit
        # SWAP on top: inherit previous non-SWAP card's suit/rank
        if is_swap(top):
            top = self._effective_top()
            if is_swap(top):
                return True  # all swaps, match anything
        return card_suit(card) == card_suit(top) or card_rank(card) == card_rank(top)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Returns (obs_next_player, reward_dict, done, info)
        reward_dict: {player_id: reward}
        """
        assert not self.done, "Game is over"

        player = self.current_player
        legal = self.legal_actions(player)
        assert action in legal, f"Illegal action {action}. Legal: {legal}"

        info = {}
        rewards = {i: 0.0 for i in range(self.num_players)}

        # --- Choose suit phase ---
        if self.phase == "choose_suit":
            self.active_suit = action - 56
            self.phase = "play"

            # Now resolve K if pending
            if self._pending_k:
                self._resolve_k(player)
                self._pending_k = False

            # Advance to next player
            self._advance_turn()
            obs = self._get_obs(self.current_player)
            return obs, rewards, False, info

        # --- Play phase ---
        if action == DRAW_ACTION:
            drawn = self._draw_card(player)
            self._record_event(player, 2)  # drew a card
            # Card is added to hand; player keeps their turn to decide
            # whether to play it (or any other card) or pass
            self.has_drawn_this_turn = True
            obs = self._get_obs(player)
            return obs, rewards, False, info
        elif action == PASS_ACTION:
            if self.has_drawn_this_turn:
                # Drew but chose not to play — game state changed, not stalemate
                self.consecutive_passes = 0
            else:
                # Hard pass: can't draw and can't play — real stalemate signal
                self.consecutive_passes += 1
            self._record_event(player, 3)  # passed
            if self.consecutive_passes >= self.num_players:
                # Stalemate: all players passed in a row
                self.done = True
                self.winner = -1  # no winner
                # Player with fewest cards gets partial reward
                min_cards = min(len(h) for h in self.hands)
                for i in range(self.num_players):
                    if len(self.hands[i]) == min_cards:
                        rewards[i] = 0.5
                    else:
                        rewards[i] = -1.0
                obs = self._get_obs(player)
                return obs, rewards, True, {"stalemate": True}
            self._advance_turn()
            obs = self._get_obs(self.current_player)
            return obs, rewards, False, info
        else:
            return self._play_card(player, action, rewards, info)

    def _play_card(self, player: int, card: int, rewards: dict, info: dict):
        self.consecutive_passes = 0
        self._record_event(player, 0)  # played_card
        hand = self.hands[player]
        assert card in hand, f"Card {card} not in hand of player {player}"
        hand.remove(card)
        self.discard.append(card)

        # Clear active suit (unless new card is 8 or SWAP — SWAP inherits)
        if not is_swap(card):
            self.active_suit = None

        # Clear swap knowledge for this player (cards change over time)
        # We keep it until they play; after playing, knowledge decays
        # Actually let's just keep swap_known_cards until overwritten

        # Check win
        if len(hand) == 0:
            self.done = True
            self.winner = player
            rewards[player] = 1.0
            for i in range(self.num_players):
                if i != player:
                    rewards[i] = -1.0
            obs = self._get_obs(player)
            return obs, rewards, True, {"winner": player}

        # Handle special cards
        if is_swap(card):
            self._resolve_swap(player)
            self._advance_turn()
        elif card_rank(card) == RANK_8:
            # Need to choose suit
            self.phase = "choose_suit"
            self._pending_8_player = player
            # Check if K also (8 is rank 7, K is rank 12 — not the same, so no overlap)
            # 8 is not K, so no K effect here
            obs = self._get_obs(player)
            return obs, rewards, False, info
        elif card_rank(card) == RANK_K:
            # All other players draw 1
            self._resolve_k(player)
            self._advance_turn()
        elif card_rank(card) == RANK_Q:
            # Reverse direction (no effect in 2-player)
            if self.num_players > 2:
                self.direction *= -1
            self._advance_turn()
        elif card_rank(card) == RANK_J:
            # Skip next player
            self._advance_turn()  # skip
            self._advance_turn()  # to the one after
        else:
            self._advance_turn()

        obs = self._get_obs(self.current_player)
        return obs, rewards, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _advance_turn(self):
        self.has_drawn_this_turn = False
        self.current_player = (self.current_player + self.direction) % self.num_players

    def _draw_card(self, player: int) -> Optional[int]:
        if not self.deck:
            self._reshuffle_discard()
        if not self.deck:
            return None  # No cards left anywhere
        card = self.deck.pop()
        self.hands[player].append(card)
        return card

    def _reshuffle_discard(self):
        """Reshuffle all but the top card of the discard pile into the deck."""
        if len(self.discard) <= 1:
            return
        top = self.discard[-1]
        self.deck = self.discard[:-1]
        self.discard = [top]
        self.rng.shuffle(self.deck)

    def _resolve_k(self, player: int):
        """All players except `player` draw 1 card."""
        for i in range(self.num_players):
            if i != player:
                self._draw_card(i)

    def _record_event(self, player: int, event_type: int):
        """Record a visible game event."""
        self.action_history.append((player, event_type))
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

    def _resolve_swap(self, player: int):
        """Swap hands with the next player."""
        next_player = (player + self.direction) % self.num_players
        self.hands[player], self.hands[next_player] = self.hands[next_player], self.hands[player]
        # After swap, `player` now has what `next_player` had → player knows these cards
        # And `next_player` now has what `player` had → next_player knows these cards
        self.swap_known_cards[player] = list(self.hands[player])
        self.swap_known_cards[next_player] = list(self.hands[next_player])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def render(self):
        print(f"--- Turn: Player {self.current_player} | Direction: {'→' if self.direction == 1 else '←'} ---")
        top = self.discard[-1]
        suit_names = ["♠", "♥", "♦", "♣"]
        top_str = card_name(top)
        if self.active_suit is not None:
            top_str += f" (active suit: {suit_names[self.active_suit]})"
        print(f"Top card: {top_str}")
        for i in range(self.num_players):
            hand_str = ", ".join(card_name(c) for c in sorted(self.hands[i]))
            marker = " ◀" if i == self.current_player else ""
            print(f"  Player {i}: [{len(self.hands[i])}] {hand_str}{marker}")
        print(f"Deck: {len(self.deck)} cards")
        if self.done:
            print(f"🏆 Player {self.winner} wins!")

    def copy(self):
        """Return a deep copy of the environment state."""
        import copy
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = BlazingEightsEnv(num_players=3, seed=42)
    env.render()
    print()

    for step_i in range(200):
        player = env.current_player
        actions = env.legal_actions()
        if not actions:
            break
        action = env.rng.choice(actions)
        print(f"Player {player} plays: {card_name(action) if action < NUM_CARDS else ('suit ' + '♠♥♦♣'[action-56] if action < DRAW_ACTION else ('DRAW' if action == DRAW_ACTION else 'PASS'))}")
        obs, rewards, done, info = env.step(action)
        if done:
            env.render()
            break
    else:
        print("Game didn't finish in 200 steps")
