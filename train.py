"""
PPO Self-Play Training for Blazing Eights.

Architecture:
  - Single policy network shared across all seats
  - Self-play: all players use the same (latest) policy
  - Collect trajectories by running full games
  - Standard PPO update with masked invalid actions

Usage:
  python train.py --num_players 2 --episodes 100000 --save_path model.pt
  python train.py --num_players 3 --episodes 200000
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from tqdm import tqdm

from blazing_env import BlazingEightsEnv, TOTAL_ACTIONS, NUM_CARDS, DRAW_ACTION, PASS_ACTION


# ---------------------------------------------------------------------------
# Policy + Value Network
# ---------------------------------------------------------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int = 180, action_size: int = TOTAL_ACTIONS, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor):
        """
        obs: (B, obs_size)
        legal_mask: (B, action_size) — 1 for legal, 0 for illegal
        Returns: logits (masked), value
        """
        h = self.shared(obs)
        logits = self.policy_head(h)
        # Mask illegal actions with large negative
        logits = logits + (1 - legal_mask) * (-1e9)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action(self, obs: np.ndarray, legal_actions: list[int], device="cpu"):
        """Sample an action from the policy."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.zeros(1, TOTAL_ACTIONS, device=device)
        for a in legal_actions:
            mask[0, a] = 1.0
        with torch.no_grad():
            logits, value = self.forward(obs_t, mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_t: torch.Tensor, mask_t: torch.Tensor, actions_t: torch.Tensor):
        """Evaluate actions for PPO update."""
        logits, values = self.forward(obs_t, mask_t)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()
        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# Trajectory Collection
# ---------------------------------------------------------------------------
class Transition:
    __slots__ = ["obs", "action", "log_prob", "value", "reward", "done", "legal_mask"]

    def __init__(self, obs, action, log_prob, value, reward, done, legal_mask):
        self.obs = obs
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.done = done
        self.legal_mask = legal_mask


def collect_game(env: BlazingEightsEnv, model: PolicyValueNet, device="cpu"):
    """
    Play one full game, return per-player trajectories.
    All players use the same model (self-play).
    """
    obs = env.reset()
    trajectories: dict[int, list[Transition]] = defaultdict(list)
    max_steps = 500

    for _ in range(max_steps):
        player = env.current_player
        legal = env.legal_actions()
        if not legal:
            break

        action, log_prob, value = model.get_action(obs, legal, device)

        # Build legal mask
        legal_mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        for a in legal:
            legal_mask[a] = 1.0

        obs_next, rewards, done, info = env.step(action)

        # Store transition for the acting player
        trajectories[player].append(Transition(
            obs=obs.copy(),
            action=action,
            log_prob=log_prob,
            value=value,
            reward=rewards[player],
            done=done,
            legal_mask=legal_mask,
        ))

        # If done, also assign terminal rewards to other players' last transitions
        if done:
            for p in range(env.num_players):
                if p != player and trajectories[p]:
                    trajectories[p][-1].reward = rewards[p]
                    trajectories[p][-1].done = True
            break

        obs = obs_next

    return trajectories


# ---------------------------------------------------------------------------
# PPO Update
# ---------------------------------------------------------------------------
def compute_gae(transitions: list[Transition], gamma=0.99, lam=0.95):
    """Compute GAE returns and advantages."""
    T = len(transitions)
    if T == 0:
        return [], []

    rewards = [t.reward for t in transitions]
    values = [t.value for t in transitions]
    dones = [t.done for t in transitions]

    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1 or dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + np.array(values)
    return returns.tolist(), advantages.tolist()


def ppo_update(model: PolicyValueNet, optimizer: torch.optim.Optimizer,
               all_transitions: list[Transition], device="cpu",
               epochs=4, batch_size=256, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO clipped surrogate update."""
    if not all_transitions:
        return {}

    # Prepare tensors
    obs_arr = np.array([t.obs for t in all_transitions])
    actions_arr = np.array([t.action for t in all_transitions])
    old_log_probs_arr = np.array([t.log_prob for t in all_transitions])
    masks_arr = np.array([t.legal_mask for t in all_transitions])

    # Compute GAE (treat all transitions as one sequence — not ideal, but we
    # already computed per-game, so we just concatenate pre-computed values)
    returns_arr = np.array([t.reward for t in all_transitions])  # placeholder
    advantages_arr = np.array([t.reward for t in all_transitions])  # placeholder

    obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions_arr, dtype=torch.long, device=device)
    old_log_probs_t = torch.tensor(old_log_probs_arr, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks_arr, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns_arr, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages_arr, dtype=torch.float32, device=device)

    # Normalize advantages
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    total_loss_sum = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.arange(len(all_transitions))
        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            idx = indices[start:end]

            b_obs = obs_t[idx]
            b_actions = actions_t[idx]
            b_old_lp = old_log_probs_t[idx]
            b_masks = masks_t[idx]
            b_returns = returns_t[idx]
            b_advantages = advantages_t[idx]

            new_log_probs, values, entropy = model.evaluate(b_obs, b_masks, b_actions)

            ratio = torch.exp(new_log_probs - b_old_lp)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, b_returns)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return {"loss": total_loss_sum / max(n_updates, 1)}


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training for {args.num_players} players, {args.episodes} episodes")

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Stats
    win_counts = defaultdict(int)
    game_lengths = []
    batch_transitions = []

    pbar = tqdm(range(1, args.episodes + 1), desc="Training", unit="ep")
    for ep in pbar:
        env = BlazingEightsEnv(num_players=args.num_players)
        trajectories = collect_game(env, model, device)

        # Record stats
        if env.done:
            win_counts[env.winner] += 1
        game_lengths.append(sum(len(v) for v in trajectories.values()))

        # Compute GAE per player and collect
        for player, trans_list in trajectories.items():
            returns, advantages = compute_gae(trans_list, gamma=args.gamma, lam=args.lam)
            for i, t in enumerate(trans_list):
                t.reward = returns[i] if i < len(returns) else t.reward
                # Store advantage in a hacky way: overwrite reward with return,
                # and we'll use (return - value) as advantage in update
            batch_transitions.extend(trans_list)

        # Update every `update_every` episodes
        if ep % args.update_every == 0:
            # Recompute advantages from stored returns and values
            for t in batch_transitions:
                pass  # returns already in t.reward

            # Build proper advantages
            for t in batch_transitions:
                # t.reward is now the GAE return; advantage = return - value
                t.reward = t.reward  # this is the return
                # We'll set the advantage in the update
            # Actually, let's just pass returns and let update compute
            returns_for_update = np.array([t.reward for t in batch_transitions])
            values_for_update = np.array([t.value for t in batch_transitions])
            advs = returns_for_update - values_for_update

            # Overwrite for the update function
            obs_arr = np.array([t.obs for t in batch_transitions])
            actions_arr = np.array([t.action for t in batch_transitions])
            old_lp_arr = np.array([t.log_prob for t in batch_transitions])
            masks_arr = np.array([t.legal_mask for t in batch_transitions])

            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions_arr, dtype=torch.long, device=device)
            old_lp_t = torch.tensor(old_lp_arr, dtype=torch.float32, device=device)
            masks_t = torch.tensor(masks_arr, dtype=torch.float32, device=device)
            returns_t = torch.tensor(returns_for_update, dtype=torch.float32, device=device)
            advs_t = torch.tensor(advs, dtype=torch.float32, device=device)

            if len(advs_t) > 1:
                advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

            # Manual PPO update
            for _ in range(args.ppo_epochs):
                indices = np.arange(len(batch_transitions))
                np.random.shuffle(indices)
                for start in range(0, len(indices), args.batch_size):
                    end = min(start + args.batch_size, len(indices))
                    idx = indices[start:end]

                    b_obs = obs_t[idx]
                    b_actions = actions_t[idx]
                    b_old_lp = old_lp_t[idx]
                    b_masks = masks_t[idx]
                    b_returns = returns_t[idx]
                    b_advs = advs_t[idx]

                    new_lp, values, entropy = model.evaluate(b_obs, b_masks, b_actions)
                    ratio = torch.exp(new_lp - b_old_lp)
                    surr1 = ratio * b_advs
                    surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, b_returns)
                    loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            batch_transitions = []

        # Logging
        if ep % args.log_every == 0:
            avg_len = np.mean(game_lengths[-args.log_every:]) if game_lengths else 0
            total_games = sum(win_counts.values())
            wr0 = win_counts[0] / max(total_games, 1)
            pbar.set_postfix(avg_len=f"{avg_len:.1f}", wr0=f"{wr0:.1%}", games=total_games)

        # Save checkpoint
        if ep % args.save_every == 0:
            path = f"{args.save_path}_ep{ep}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep,
                "num_players": args.num_players,
            }, path)
            tqdm.write(f"  Saved checkpoint: {path}")

    # Final save
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": args.episodes,
        "num_players": args.num_players,
    }, f"{args.save_path}_final.pt")
    print(f"Training complete. Final model saved to {args.save_path}_final.pt")

    return model


# ---------------------------------------------------------------------------
# Evaluation: play against random
# ---------------------------------------------------------------------------
def evaluate_vs_random(model: PolicyValueNet, num_players=2, num_games=1000, device="cpu"):
    """Player 0 = model, others = random. Returns player 0 win rate."""
    wins = 0
    for _ in range(num_games):
        env = BlazingEightsEnv(num_players=num_players)
        obs = env.reset()
        for _ in range(500):
            player = env.current_player
            legal = env.legal_actions()
            if not legal:
                break
            if player == 0:
                action, _, _ = model.get_action(obs, legal, device)
            else:
                action = np.random.choice(legal)
            obs, rewards, done, info = env.step(action)
            if done:
                if env.winner == 0:
                    wins += 1
                break
    return wins / num_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Blazing Eights PPO agent")
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_every", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_path", type=str, default="blazing_ppo")
    args = parser.parse_args()

    model = train(args)

    # Eval vs random
    print("\nEvaluating vs random opponents...")
    for n in [2, 3, 4, 5]:
        if n <= args.num_players + 1:  # only eval for trained player count
            wr = evaluate_vs_random(model, num_players=n, num_games=1000)
            print(f"  {n} players: win rate = {wr:.1%} (random baseline: {1/n:.1%})")
