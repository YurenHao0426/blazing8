"""
Blazing Eights — Human vs AI interactive game.

Play against the trained PPO agent in your terminal.

Usage:
  python versus.py --model blazing_ppo_final.pt
  python versus.py --model blazing_ppo_final.pt --num_players 3  # you + 2 AI
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from blazing_env import (
    BlazingEightsEnv, card_name, card_suit, card_rank,
    is_swap, RANK_8, RANK_J, RANK_Q, RANK_K,
    NUM_CARDS, TOTAL_ACTIONS, DRAW_ACTION, PASS_ACTION,
)
from train import PolicyValueNet

AI_COLOR = "\033[91m"   # red for AI actions
AI_RESET = "\033[0m"
# Suit colors: ♠blue ♥magenta ♦yellow ♣cyan
SUIT_COLORS = ["\033[94m", "\033[35m", "\033[93m", "\033[96m"]

SUIT_SYMBOLS = ["♠", "♥", "♦", "♣"]
SUIT_LETTERS = {"s": 0, "h": 1, "d": 2, "c": 3,
                "♠": 0, "♥": 1, "♦": 2, "♣": 3}
RANK_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def card_effect(c: int, num_players: int = 2) -> str:
    """Return a short effect tag for special cards."""
    if is_swap(c):
        return "\033[93m换牌\033[0m"
    r = card_rank(c)
    if r == RANK_8:
        return "\033[93m万能\033[0m"
    if r == RANK_K:
        return "\033[93m全摸\033[0m"
    if r == RANK_Q:
        return "\033[93m反转\033[0m" if num_players > 2 else ""
    if r == RANK_J:
        return "\033[93m跳过\033[0m"
    return ""


def pretty_card(c: int) -> str:
    if is_swap(c):
        return "\033[95mSWAP\033[0m"
    suit = card_suit(c)
    rank = RANK_NAMES[card_rank(c)]
    return f"{SUIT_COLORS[suit]}{rank}{SUIT_SYMBOLS[suit]}\033[0m"


def pretty_hand(hand: list[int], num_players: int = 2) -> str:
    sorted_hand = sorted(hand, key=lambda c: (card_suit(c) if not is_swap(c) else 99, c))
    parts = []
    for i, c in enumerate(sorted_hand):
        effect = card_effect(c, num_players)
        tag = f"[{i}] {pretty_card(c)}"
        if effect:
            tag += f"({effect})"
        parts.append(tag)
    return "  ".join(parts)


def print_game_state(env: BlazingEightsEnv, human_player: int, show_ai_hand: bool = False):
    print()
    print("=" * 55)
    top = env.discard[-1]
    top_str = pretty_card(top)
    if env.active_suit is not None:
        s = env.active_suit
        top_str += f"  (指定花色: {SUIT_COLORS[s]}{SUIT_SYMBOLS[s]}{AI_RESET})"
    dir_str = "顺时针 →" if env.direction == 1 else "逆时针 ←"
    print(f"  弃牌堆顶: {top_str}    方向: {dir_str}    牌堆剩余: {len(env.deck)}")
    print("-" * 55)
    for i in range(env.num_players):
        if i == human_player:
            tag = "你"
            hand_str = f"{len(env.hands[i])} 张牌"
        else:
            tag = f"AI-{i}"
            if show_ai_hand:
                hand_str = ", ".join(pretty_card(c) for c in sorted(env.hands[i]))
            else:
                hand_str = f"{len(env.hands[i])} 张牌"
        arrow = " ◀" if i == env.current_player else ""
        print(f"  {tag}: {hand_str}{arrow}")
    print("=" * 55)


def parse_card_input(s: str) -> int:
    s = s.strip().upper()
    if s.startswith("SWAP") or s.startswith("SW"):
        return 52
    if s.startswith("10"):
        rank_str, suit_str = "10", s[2:].lower()
    else:
        rank_str, suit_str = s[0], s[1:].lower()
    rank_map = {r: i for i, r in enumerate(RANK_NAMES)}
    if rank_str not in rank_map or suit_str not in SUIT_LETTERS:
        raise ValueError(f"无法识别: {s}  (格式例: 8h, Ks, 10d, Ac, SWAP)")
    return SUIT_LETTERS[suit_str] * 13 + rank_map[rank_str]


def human_choose_action(env: BlazingEightsEnv, player: int) -> int:
    hand = sorted(env.hands[player], key=lambda c: (card_suit(c) if not is_swap(c) else 99, c))
    legal = env.legal_actions(player)

    if env.phase == "choose_suit":
        print("\n  你打出了 8！选择指定花色:")
        for i, s in enumerate(SUIT_SYMBOLS):
            print(f"    [{i}] {SUIT_COLORS[i]}{s}{AI_RESET}")
        while True:
            try:
                choice = input("  选择 (0-3): ").strip()
                idx = int(choice)
                action = 56 + idx
                if action in legal:
                    return action
                print("  无效选择，请重试")
            except (ValueError, IndexError):
                print("  请输入 0-3")
        return action

    print(f"\n  你的手牌: {pretty_hand(hand, env.num_players)}")

    # Build playable cards display
    playable = [a for a in legal if a < NUM_CARDS]
    can_draw = DRAW_ACTION in legal
    can_pass = PASS_ACTION in legal

    print("  可出的牌:", end="")
    if playable:
        playable_names = []
        for a in playable:
            idx_in_hand = hand.index(a)
            effect = card_effect(a, env.num_players)
            tag = f"[{idx_in_hand}]{pretty_card(a)}"
            if effect:
                tag += f"({effect})"
            playable_names.append(tag)
        print("  " + "  ".join(playable_names))
    else:
        print("  无")

    if can_draw:
        print("  [d] 摸牌")
    if can_pass:
        if env.has_drawn_this_turn and not playable:
            print("  [p] 跳过 (无法出牌)")
        elif env.has_drawn_this_turn:
            print("  [p] 不出牌")
        else:
            print("  [p] 跳过 (牌堆已空)")

    while True:
        choice = input("  你的选择: ").strip().lower()
        if choice == "d" and can_draw:
            return DRAW_ACTION
        if choice == "p" and can_pass:
            return PASS_ACTION
        if choice == "d" and not can_draw:
            print("  牌堆已空，无法摸牌")
            continue
        if choice == "p" and not can_pass:
            print("  还没摸牌，不能直接跳过")
            continue
        if choice == "q":
            raise KeyboardInterrupt
        try:
            idx = int(choice)
            if 0 <= idx < len(hand):
                card = hand[idx]
                if card in playable:
                    return card
                # Handle swap cards (might have multiple)
                if is_swap(card):
                    for a in playable:
                        if is_swap(a):
                            return a
                print(f"  {pretty_card(card)} 不能出，请选其他牌")
            else:
                print(f"  序号超出范围 (0-{len(hand)-1})")
        except ValueError:
            print("  输入序号、d(摸牌) 或 q(退出)")


def ai_choose_action(env: BlazingEightsEnv, model: PolicyValueNet, player: int, device="cpu") -> int:
    obs = env._get_obs(player)
    legal = env.legal_actions(player)
    action, _, value = model.get_action(obs, legal, device)
    return action


def describe_action(player_name: str, action: int, env: BlazingEightsEnv, drawn_card: int = None):
    if action == DRAW_ACTION:
        return f"  {player_name} 摸了一张牌"
    if action == PASS_ACTION:
        return f"  {player_name} 跳过"
    if action >= 56:
        suit = action - 56
        return f"  {player_name} 指定花色: {SUIT_COLORS[suit]}{SUIT_SYMBOLS[suit]}{AI_RESET}"
    desc = f"  {player_name} 打出 {pretty_card(action)}"
    rank = card_rank(action)
    if is_swap(action):
        desc += "  → 与下家交换手牌！"
    elif rank == RANK_8:
        desc += "  → 万能牌！选择花色..."
    elif rank == RANK_K:
        desc += "  → 其他所有人各摸 1 张！"
    elif rank == RANK_Q and env.num_players > 2:
        desc += "  → 反转方向！"
    elif rank == RANK_J:
        desc += "  → 跳过下一位！"
    return desc


def play_game(model_path: str, num_players: int, human_player: int = 0, show_ai: bool = False):
    device = "cpu"
    model = PolicyValueNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print()
    print("╔══════════════════════════════════════╗")
    print("║     Blazing Eights - 人机对战        ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  玩家数: {num_players}   你是: Player {human_player}         ║")
    print("║  输入序号出牌, d摸牌, p跳过, q退出   ║")
    print("╚══════════════════════════════════════╝")

    env = BlazingEightsEnv(num_players=num_players)
    turn = 0

    while not env.done:
        player = env.current_player
        turn += 1

        if player == human_player:
            print_game_state(env, human_player, show_ai_hand=show_ai)
            try:
                action = human_choose_action(env, player)
            except KeyboardInterrupt:
                print("\n\n  你退出了游戏。再见！")
                return

            # Describe human action
            name = "你"
            if action == DRAW_ACTION:
                # Remember hand before draw to find the new card
                hand_before = set(env.hands[player])
                obs, rewards, done, info = env.step(action)
                hand_after = set(env.hands[player])
                new_cards = hand_after - hand_before
                if new_cards:
                    drawn = next(iter(new_cards))
                    print(f"  你摸到了 {pretty_card(drawn)}")
                else:
                    print(f"  牌堆已空，没摸到牌")
                # Turn stays with human — loop back to let them decide
                continue
            elif action == PASS_ACTION:
                print(f"  你选择不出牌，结束回合")
                obs, rewards, done, info = env.step(action)
                continue
            else:
                print(describe_action(name, action, env))
                obs, rewards, done, info = env.step(action)
                # If played an 8, need to choose suit
                if env.phase == "choose_suit" and env._pending_8_player == human_player:
                    suit_action = human_choose_action(env, human_player)
                    si = suit_action - 56
                    print(f"  你指定花色: {SUIT_COLORS[si]}{SUIT_SYMBOLS[si]}{AI_RESET}")
                    obs, rewards, done, info = env.step(suit_action)
                continue
        else:
            # AI turn
            ai_name = f"AI-{player}"

            if env.phase == "choose_suit":
                action = ai_choose_action(env, model, player, device)
                si = action - 56
                print(f"  {AI_COLOR}{ai_name} 指定花色: {SUIT_COLORS[si]}{SUIT_SYMBOLS[si]}{AI_RESET}")
                obs, rewards, done, info = env.step(action)
                continue

            action = ai_choose_action(env, model, player, device)

            if action == DRAW_ACTION:
                obs, rewards, done, info = env.step(action)
                # Check if AI has playable cards after drawing (observable "tell")
                ai_legal = env.legal_actions(player)
                has_playable = any(a < NUM_CARDS or (56 <= a <= 59) for a in ai_legal)
                if has_playable:
                    print(f"  {AI_COLOR}{ai_name} 摸了一张牌 (有牌可出){AI_RESET}")
                else:
                    print(f"  {AI_COLOR}{ai_name} 摸了一张牌 (无牌可出){AI_RESET}")
                # AI still has their turn — now decide to play or pass
                action2 = ai_choose_action(env, model, player, device)
                if action2 == PASS_ACTION:
                    print(f"  {AI_COLOR}{ai_name} 选择不出牌{AI_RESET}")
                    obs, rewards, done, info = env.step(action2)
                else:
                    print(f"  {AI_COLOR}{describe_action(ai_name, action2, env).strip()}{AI_RESET}")
                    obs, rewards, done, info = env.step(action2)
                    if env.phase == "choose_suit" and env._pending_8_player == player:
                        suit_action = ai_choose_action(env, model, player, device)
                        si = suit_action - 56
                        print(f"  {AI_COLOR}{ai_name} 指定花色: {SUIT_COLORS[si]}{SUIT_SYMBOLS[si]}{AI_RESET}")
                        obs, rewards, done, info = env.step(suit_action)
            elif action == PASS_ACTION:
                print(f"  {AI_COLOR}{ai_name} 跳过{AI_RESET}")
                obs, rewards, done, info = env.step(action)
            else:
                print(f"  {AI_COLOR}{describe_action(ai_name, action, env).strip()}{AI_RESET}")
                obs, rewards, done, info = env.step(action)
                if env.phase == "choose_suit" and env._pending_8_player == player:
                    suit_action = ai_choose_action(env, model, player, device)
                    si = suit_action - 56
                    print(f"  {AI_COLOR}{ai_name} 指定花色: {SUIT_COLORS[si]}{SUIT_SYMBOLS[si]}{AI_RESET}")
                    obs, rewards, done, info = env.step(suit_action)

    # Game over
    print_game_state(env, human_player, show_ai_hand=True)
    print()
    if env.winner == human_player:
        print("  🎉 你赢了！！！")
    elif env.winner >= 0:
        print(f"  💀 AI-{env.winner} 赢了...")
    else:
        print("  平局（僵局）")

    # Show hand sizes
    for i in range(env.num_players):
        name = "你" if i == human_player else f"AI-{i}"
        print(f"    {name}: {len(env.hands[i])} 张剩余")
    print()


def main():
    parser = argparse.ArgumentParser(description="Blazing Eights 人机对战")
    parser.add_argument("--model", type=str, default="blazing_ppo_final.pt", help="模型路径")
    parser.add_argument("--num_players", type=int, default=2, help="玩家总数 (2-5)")
    parser.add_argument("--show_ai", action="store_true", help="显示 AI 手牌 (调试用)")
    args = parser.parse_args()

    while True:
        play_game(args.model, args.num_players, human_player=0, show_ai=args.show_ai)
        again = input("  再来一局? (y/n): ").strip().lower()
        if again != "y":
            print("  下次再见！")
            break


if __name__ == "__main__":
    main()
