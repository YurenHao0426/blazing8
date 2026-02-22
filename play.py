"""
Blazing Eights — Real-time Play Assistant.

Load a trained model and get recommended actions during a real game.
You input the game state, it tells you the best move.

Usage:
  python play.py --model blazing_ppo_final.pt --num_players 3
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from blazing_env import (
    BlazingEightsEnv, PolicyValueNet, card_name, card_suit, card_rank,
    is_swap, RANK_8, RANK_J, RANK_Q, RANK_K, NUM_STANDARD, NUM_CARDS,
    TOTAL_ACTIONS, DRAW_ACTION, PASS_ACTION
)

# Import network
import sys
sys.path.insert(0, ".")
from train import PolicyValueNet


SUIT_NAMES = ["♠ spades", "♥ hearts", "♦ diamonds", "♣ clubs"]
SUIT_SHORT = ["s", "h", "d", "c"]
RANK_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def parse_card(s: str) -> int:
    """Parse a card string like '8h', 'Ks', 'SWAP', '10d' into card index."""
    s = s.strip().upper()
    if s.startswith("SWAP"):
        # We don't distinguish between swap cards; just return first available
        return 52  # caller should handle
    if s.startswith("SW"):
        return 52

    # Parse rank
    if s.startswith("10"):
        rank_str = "10"
        suit_str = s[2:].lower()
    else:
        rank_str = s[0]
        suit_str = s[1:].lower()

    rank_map = {r: i for i, r in enumerate(RANK_NAMES)}
    suit_map = {"s": 0, "h": 1, "d": 2, "c": 3,
                "♠": 0, "♥": 1, "♦": 2, "♣": 3}

    if rank_str not in rank_map or suit_str not in suit_map:
        raise ValueError(f"Cannot parse card: {s}")

    return suit_map[suit_str] * 13 + rank_map[rank_str]


def build_obs_from_input(hand: list[int], top_card: int, active_suit: int | None,
                         direction: int, other_hand_sizes: list[int],
                         deck_size: int, num_players: int,
                         known_opponent_cards: list[int] | None = None,
                         other_last_events: list[int] | None = None,
                         other_draw_streaks: list[int] | None = None) -> np.ndarray:
    """Build observation vector from manual game state input."""
    obs = np.zeros(180, dtype=np.float32)

    # Hand
    for c in hand:
        obs[c] = 1.0

    # Top card suit
    if active_suit is not None:
        suit = active_suit
    elif not is_swap(top_card):
        suit = card_suit(top_card)
    else:
        suit = 0
    obs[56 + suit] = 1.0

    # Top card rank
    if not is_swap(top_card) and active_suit is None:
        obs[60 + card_rank(top_card)] = 1.0

    # Direction
    obs[73] = 0.0 if direction == 1 else 1.0

    # Other players' hand sizes
    for i, sz in enumerate(other_hand_sizes):
        obs[74 + i] = sz / 20.0

    # Deck size
    obs[74 + num_players - 1] = deck_size / 56.0

    # Phase (always play in interactive mode)
    obs[75 + num_players - 1] = 0.0

    # Known opponent cards
    if known_opponent_cards:
        offset = 76 + num_players - 1
        for c in known_opponent_cards:
            obs[offset + c] = 1.0

    # Per other player draw info
    draw_info_offset = 132 + num_players - 1
    if other_last_events:
        for i, evt in enumerate(other_last_events):
            if evt >= 0:
                obs[draw_info_offset + i * 5 + evt] = 1.0
    if other_draw_streaks:
        for i, streak in enumerate(other_draw_streaks):
            obs[draw_info_offset + i * 5 + 4] = streak / 10.0

    return obs


def get_recommendations(model: PolicyValueNet, obs: np.ndarray, hand: list[int],
                        top_card: int, active_suit: int | None, device="cpu"):
    """Get action probabilities and recommendations."""
    # Determine legal actions
    legal = []
    for c in hand:
        if is_swap(c):
            legal.append(c)
        elif card_rank(c) == RANK_8:
            legal.append(c)
        elif active_suit is not None:
            if card_suit(c) == active_suit:
                legal.append(c)
        elif not is_swap(top_card):
            if card_suit(c) == card_suit(top_card) or card_rank(c) == card_rank(top_card):
                legal.append(c)

    if not legal:
        legal = [DRAW_ACTION]  # Caller should check deck; in practice use env's legal_actions

    # Get model probabilities
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.zeros(1, TOTAL_ACTIONS, device=device)
    for a in legal:
        mask[0, a] = 1.0

    with torch.no_grad():
        logits, value = model.forward(obs_t, mask)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # Sort by probability
    ranked = []
    for a in legal:
        if a == DRAW_ACTION:
            name = "DRAW"
        elif a >= NUM_CARDS:
            name = f"Choose {SUIT_NAMES[a - 56]}"
        else:
            name = card_name(a)
        ranked.append((a, name, probs[a]))

    ranked.sort(key=lambda x: -x[2])
    return ranked, value.item()


def interactive_loop(model_path: str, num_players: int):
    device = "cpu"

    # Load model
    model = PolicyValueNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded model from {model_path}")
    print(f"Trained for {checkpoint.get('episode', '?')} episodes, "
          f"{checkpoint.get('num_players', '?')} players")
    print()

    print("=" * 60)
    print("  Blazing Eights — Play Assistant")
    print("  Card format: rank+suit (e.g., 8h, Ks, 10d, Ac, SWAP)")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        print("\n--- New Turn ---")
        try:
            # Hand
            hand_str = input("Your hand (comma-separated, e.g., 8h,Ks,3d,SWAP): ").strip()
            if hand_str.lower() == "quit":
                break
            hand = [parse_card(c) for c in hand_str.split(",")]

            # Top card
            top_str = input("Top card on discard pile: ").strip()
            top_card = parse_card(top_str)

            # Active suit (if top is 8)
            active_suit = None
            if card_rank(top_card) == RANK_8:
                suit_str = input("Active suit (s/h/d/c): ").strip().lower()
                suit_map = {"s": 0, "h": 1, "d": 2, "c": 3}
                active_suit = suit_map.get(suit_str)

            # Direction
            dir_str = input("Direction (cw/ccw) [cw]: ").strip().lower()
            direction = -1 if dir_str == "ccw" else 1

            # Other players' hand sizes
            sizes_str = input(f"Other players' hand sizes (comma-sep, {num_players-1} values): ").strip()
            other_sizes = [int(x) for x in sizes_str.split(",")]

            # Deck size estimate
            deck_str = input("Approximate deck size [20]: ").strip()
            deck_size = int(deck_str) if deck_str else 20

            # Draw info for other players
            # p=played from hand, d=drew and played, s=drew and skipped, ?=unknown
            event_str = input(f"Other players' last action ({num_players-1} values, p/d/s/?): ").strip().lower()
            event_map = {"p": 0, "d": 1, "s": 2, "?": -1, "": -1}
            other_events = None
            if event_str:
                other_events = [event_map.get(x.strip(), -1) for x in event_str.split(",")]

            streak_str = input(f"Other players' consecutive draw-skip count ({num_players-1} values) [0s]: ").strip()
            other_streaks = None
            if streak_str:
                other_streaks = [int(x) for x in streak_str.split(",")]

            # Build obs and get recommendation
            obs = build_obs_from_input(
                hand, top_card, active_suit, direction,
                other_sizes, deck_size, num_players,
                other_last_events=other_events,
                other_draw_streaks=other_streaks,
            )
            ranked, value = get_recommendations(model, obs, hand, top_card, active_suit, device)

            print(f"\n  Win probability estimate: {(value + 1) / 2:.1%}")
            print("  Recommended actions:")
            for i, (action, name, prob) in enumerate(ranked):
                bar = "█" * int(prob * 30)
                print(f"    {'→' if i == 0 else ' '} {name:<12s}  {prob:6.1%}  {bar}")

            # If best action is an 8, also show suit recommendation
            if ranked and ranked[0][0] < NUM_CARDS and card_rank(ranked[0][0]) == RANK_8:
                print("\n  If you play 8, recommended suit:")
                # Quick eval for each suit
                for suit_idx in range(4):
                    temp_obs = obs.copy()
                    # Set active suit
                    temp_obs[56:60] = 0
                    temp_obs[56 + suit_idx] = 1.0
                    temp_obs[60:73] = 0  # clear rank (wild)
                    obs_t = torch.tensor(temp_obs, dtype=torch.float32).unsqueeze(0)
                    mask = torch.ones(1, TOTAL_ACTIONS)  # dummy
                    with torch.no_grad():
                        _, v = model.forward(obs_t, mask)
                    print(f"    {SUIT_NAMES[suit_idx]}: estimated value {v.item():.3f}")

        except (ValueError, IndexError) as e:
            print(f"  Error: {e}. Try again.")
        except KeyboardInterrupt:
            break

    print("Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--num_players", type=int, default=2)
    args = parser.parse_args()
    interactive_loop(args.model, args.num_players)
