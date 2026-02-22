"""
PPO Self-Play Training for Blazing Eights.

Architecture:
  - Single policy network shared across all seats
  - Self-play: all players use the same (latest) policy
  - Batched game collection: many games run in parallel with batched inference
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
# Trajectory Storage
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


def greedy_random_action(legal: list[int]) -> int:
    """Pick a random playable card; only draw/pass if no card to play."""
    play_actions = [a for a in legal if a < NUM_CARDS or (56 <= a <= 59)]
    if play_actions:
        return int(np.random.choice(play_actions))
    return int(np.random.choice(legal))


# ---------------------------------------------------------------------------
# Batched Game Collection
# ---------------------------------------------------------------------------
def collect_games_batch(num_games: int, num_players: int, model: PolicyValueNet,
                        device="cpu", max_steps=500):
    """Run multiple games simultaneously with batched model inference.

    Instead of running games one-by-one (each step = batch_size=1 forward pass),
    this runs all games in lockstep: at each step, all active games' observations
    are batched into a single forward pass.

    Returns:
        envs: list of completed environments (for reading winner/done)
        trajectories: list of per-player trajectory dicts
    """
    envs = [BlazingEightsEnv(num_players=num_players) for _ in range(num_games)]
    obs_list = [env.reset() for env in envs]
    trajectories = [defaultdict(list) for _ in range(num_games)]
    active = set(range(num_games))

    for _ in range(max_steps):
        if not active:
            break

        # Gather observations and legal masks for all active games
        indices = []
        batch_obs = []
        batch_masks = []
        batch_players = []

        for i in sorted(active):
            legal = envs[i].legal_actions()
            if not legal:
                active.discard(i)
                continue
            mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
            for a in legal:
                mask[a] = 1.0
            indices.append(i)
            batch_obs.append(obs_list[i])
            batch_masks.append(mask)
            batch_players.append(envs[i].current_player)

        if not indices:
            break

        # Single batched forward pass for all active games
        obs_t = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        mask_t = torch.tensor(np.array(batch_masks), dtype=torch.float32, device=device)

        with torch.inference_mode():
            logits, values = model(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        # Step each environment
        for j, i in enumerate(indices):
            player = batch_players[j]
            action = int(actions_np[j])
            obs_next, rewards, done, info = envs[i].step(action)

            trajectories[i][player].append(Transition(
                obs=batch_obs[j],
                action=action,
                log_prob=float(log_probs_np[j]),
                value=float(values_np[j]),
                reward=rewards[player],
                done=done,
                legal_mask=batch_masks[j],
            ))

            if done:
                for p in range(envs[i].num_players):
                    if p != player and trajectories[i][p]:
                        trajectories[i][p][-1].reward = rewards[p]
                        trajectories[i][p][-1].done = True
                active.discard(i)
            else:
                obs_list[i] = obs_next

    return envs, trajectories


# ---------------------------------------------------------------------------
# PPO Utilities
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


# ---------------------------------------------------------------------------
# Greedy Warmup (Behavioral Cloning)
# ---------------------------------------------------------------------------
def greedy_warmup(model: PolicyValueNet, optimizer: torch.optim.Optimizer,
                  num_players: int, num_games: int = 2000, epochs: int = 5,
                  batch_size: int = 256, device: str = "cpu"):
    """Pre-train the model to imitate greedy play (play if possible, else draw)."""
    print(f"Greedy warmup: {num_games} games, {epochs} epochs...")
    obs_list, action_list, mask_list = [], [], []

    for _ in tqdm(range(num_games), desc="Collecting greedy data", unit="game"):
        env = BlazingEightsEnv(num_players=num_players)
        obs = env.reset()
        for _ in range(500):
            legal = env.legal_actions()
            if not legal:
                break
            action = greedy_random_action(legal)
            legal_mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
            for a in legal:
                legal_mask[a] = 1.0
            obs_list.append(obs.copy())
            action_list.append(action)
            mask_list.append(legal_mask)
            obs, _, done, _ = env.step(action)
            if done:
                break

    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(action_list), dtype=torch.long, device=device)
    mask_t = torch.tensor(np.array(mask_list), dtype=torch.float32, device=device)
    print(f"  Collected {len(obs_list)} transitions")

    for epoch in range(epochs):
        indices = np.arange(len(obs_list))
        np.random.shuffle(indices)
        total_loss = 0
        n_batches = 0
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            logits, _ = model(obs_t[idx], mask_t[idx])
            loss = F.cross_entropy(logits, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f}")


# ---------------------------------------------------------------------------
# Auto-calibrate collect_batch
# ---------------------------------------------------------------------------
def calibrate_collect_batch(num_players: int, model: PolicyValueNet, device="cpu"):
    """Find optimal parallel game batch size by benchmarking throughput.

    Strategy: test increasing batch sizes, pick the smallest one whose
    throughput is within 10% of the peak. Smaller batches mean more frequent
    PPO updates which is better for training quality.
    """
    import time
    candidates = [64, 128, 256, 512]
    rates = []

    print("Auto-calibrating collect_batch...")
    for size in candidates:
        t0 = time.time()
        collect_games_batch(size, num_players, model, device)
        elapsed = time.time() - t0
        rate = size / elapsed
        rates.append(rate)
        print(f"  batch={size}: {rate:.0f} games/s")

    peak = max(rates)
    threshold = peak * 0.9  # within 10% of peak
    for size, rate in zip(candidates, rates):
        if rate >= threshold:
            print(f"  Selected: {size} ({rate:.0f} games/s, peak={peak:.0f})")
            return size

    return candidates[-1]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(args):
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    collect_device = "cpu"  # env simulation always on CPU
    print(f"Train device: {train_device}, Collect device: {collect_device}")
    print(f"Training for {args.num_players} players, {args.episodes} episodes")

    # Model lives on CPU for game collection; moves to GPU for PPO updates
    model = PolicyValueNet().to(collect_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Greedy warmup: imitate greedy play before self-play
    if args.greedy_warmup > 0:
        if train_device != collect_device:
            model.to(train_device)
        greedy_warmup(model, optimizer, args.num_players,
                      num_games=args.greedy_warmup, device=train_device)
        if train_device != collect_device:
            model.to(collect_device)

    # Determine collect_batch size
    if args.collect_batch is not None:
        collect_batch = args.collect_batch
        print(f"Batch collection: {collect_batch} games per batch")
    else:
        collect_batch = calibrate_collect_batch(args.num_players, model, collect_device)

    # Training log
    log_path = f"{args.save_path}_log.csv"
    with open(log_path, "w") as f:
        f.write("episode,avg_len,loss,vs_greedy_wr\n")

    # Stats
    win_counts = defaultdict(int)
    all_game_lengths = []
    recent_loss = 0.0
    recent_loss_count = 0

    # Entropy annealing: start high to escape greedy local minimum, decay to target
    ent_start = args.ent_start if args.ent_start is not None else args.ent_coef * 5
    ent_end = args.ent_coef
    if ent_start != ent_end:
        print(f"Entropy annealing: {ent_start} → {ent_end}")

    ep = 0
    next_log = args.log_every
    next_eval = args.eval_every
    next_save = args.save_every
    pbar = tqdm(total=args.episodes, desc="Training", unit="ep")

    while ep < args.episodes:
        games_this_batch = min(collect_batch, args.episodes - ep)

        # Collect games in parallel with batched inference
        envs, batch_trajectories = collect_games_batch(
            games_this_batch, args.num_players, model, collect_device
        )

        # Process trajectories: compute GAE and collect all transitions
        batch_transitions = []
        for i in range(games_this_batch):
            env = envs[i]
            traj = batch_trajectories[i]
            if env.done:
                win_counts[env.winner] += 1
            all_game_lengths.append(sum(len(v) for v in traj.values()))

            for player, trans_list in traj.items():
                returns, advantages = compute_gae(trans_list, gamma=args.gamma, lam=args.lam)
                for k, t in enumerate(trans_list):
                    t.reward = returns[k] if k < len(returns) else t.reward
                batch_transitions.extend(trans_list)

        ep += games_this_batch
        pbar.update(games_this_batch)

        # PPO update
        if batch_transitions:
            returns_for_update = np.array([t.reward for t in batch_transitions])
            values_for_update = np.array([t.value for t in batch_transitions])
            advs = returns_for_update - values_for_update

            obs_arr = np.array([t.obs for t in batch_transitions])
            actions_arr = np.array([t.action for t in batch_transitions])
            old_lp_arr = np.array([t.log_prob for t in batch_transitions])
            masks_arr = np.array([t.legal_mask for t in batch_transitions])

            # Move model to train device for PPO update
            if train_device != collect_device:
                model.to(train_device)

            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=train_device)
            actions_t = torch.tensor(actions_arr, dtype=torch.long, device=train_device)
            old_lp_t = torch.tensor(old_lp_arr, dtype=torch.float32, device=train_device)
            masks_t = torch.tensor(masks_arr, dtype=torch.float32, device=train_device)
            returns_t = torch.tensor(returns_for_update, dtype=torch.float32, device=train_device)
            advs_t = torch.tensor(advs, dtype=torch.float32, device=train_device)

            if len(advs_t) > 1:
                advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

            # PPO clipped surrogate update
            batch_loss = 0.0
            n_updates = 0
            for _ in range(args.ppo_epochs):
                perm = np.arange(len(batch_transitions))
                np.random.shuffle(perm)
                for start in range(0, len(perm), args.batch_size):
                    end = min(start + args.batch_size, len(perm))
                    idx = perm[start:end]

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
                    # Entropy coefficient with linear annealing
                    progress = min(ep / args.episodes, 1.0)
                    ent_coef = ent_start + (ent_end - ent_start) * progress
                    loss = policy_loss + 0.5 * value_loss - ent_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    batch_loss += loss.item()
                    n_updates += 1

            recent_loss += batch_loss / max(n_updates, 1)
            recent_loss_count += 1

            # Move model back to CPU for collection
            if train_device != collect_device:
                model.to(collect_device)

        # Logging
        if ep >= next_log:
            avg_len = np.mean(all_game_lengths[-args.log_every:]) if all_game_lengths else 0
            avg_loss = recent_loss / max(recent_loss_count, 1)
            total_games = sum(win_counts.values())
            wr0 = win_counts[0] / max(total_games, 1)
            cur_ent = ent_start + (ent_end - ent_start) * min(ep / args.episodes, 1.0)
            pbar.set_postfix(avg_len=f"{avg_len:.1f}", loss=f"{avg_loss:.3f}",
                             ent=f"{cur_ent:.3f}", wr0=f"{wr0:.1%}")
            recent_loss = 0.0
            recent_loss_count = 0
            next_log += args.log_every

        # Evaluate vs greedy + write log
        if ep >= next_eval:
            avg_len = np.mean(all_game_lengths[-args.eval_every:]) if all_game_lengths else 0
            avg_loss_log = recent_loss / max(recent_loss_count, 1) if recent_loss_count > 0 else 0
            vs_wr = evaluate_vs_greedy_batch(model, num_players=args.num_players,
                                             num_games=500, device=collect_device)
            with open(log_path, "a") as f:
                f.write(f"{ep},{avg_len:.1f},{avg_loss_log:.4f},{vs_wr:.4f}\n")
            tqdm.write(f"  [Eval ep{ep}] avg_len={avg_len:.1f} vs_greedy={vs_wr:.1%}")
            next_eval += args.eval_every

        # Save checkpoint
        if ep >= next_save:
            path = f"{args.save_path}_ep{ep}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep,
                "num_players": args.num_players,
            }, path)
            tqdm.write(f"  Saved checkpoint: {path}")
            next_save += args.save_every

    # Final save
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": args.episodes,
        "num_players": args.num_players,
    }, f"{args.save_path}_final.pt")
    print(f"Training complete. Final model saved to {args.save_path}_final.pt")
    print(f"Training log saved to {log_path}")

    return model


# ---------------------------------------------------------------------------
# Evaluation: play against greedy (batched)
# ---------------------------------------------------------------------------
def evaluate_vs_greedy_batch(model: PolicyValueNet, num_players=2, num_games=500, device="cpu"):
    """Batched evaluation: player 0 = model, others = greedy random."""
    envs = [BlazingEightsEnv(num_players=num_players) for _ in range(num_games)]
    obs_list = [env.reset() for env in envs]
    active = set(range(num_games))

    for _ in range(500):
        if not active:
            break

        # Separate model-controlled (player 0) and greedy-controlled turns
        model_idx = []
        model_obs = []
        model_masks = []
        greedy_pairs = []

        for i in sorted(active):
            legal = envs[i].legal_actions()
            if not legal:
                active.discard(i)
                continue
            if envs[i].current_player == 0:
                mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
                for a in legal:
                    mask[a] = 1.0
                model_idx.append(i)
                model_obs.append(obs_list[i])
                model_masks.append(mask)
            else:
                greedy_pairs.append((i, greedy_random_action(legal)))

        # Batched model inference for player 0 turns
        if model_obs:
            obs_t = torch.tensor(np.array(model_obs), dtype=torch.float32, device=device)
            mask_t = torch.tensor(np.array(model_masks), dtype=torch.float32, device=device)
            with torch.inference_mode():
                logits, _ = model(obs_t, mask_t)
                actions = Categorical(F.softmax(logits, dim=-1)).sample().cpu().numpy()
            for j, i in enumerate(model_idx):
                obs_next, _, done, _ = envs[i].step(int(actions[j]))
                if done:
                    active.discard(i)
                else:
                    obs_list[i] = obs_next

        # Greedy actions for other players
        for i, action in greedy_pairs:
            obs_next, _, done, _ = envs[i].step(action)
            if done:
                active.discard(i)
            else:
                obs_list[i] = obs_next

    return sum(1 for e in envs if e.done and e.winner == 0) / num_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Blazing Eights PPO agent")
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Final entropy coefficient")
    parser.add_argument("--ent_start", type=float, default=None,
                        help="Initial entropy coef, linearly decays to ent_coef (default: 5x ent_coef)")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_every", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=2500)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_path", type=str, default="blazing_ppo")
    parser.add_argument("--collect_batch", type=int, default=None,
                        help="Parallel game collection batch size (default: same as update_every)")
    parser.add_argument("--greedy_warmup", type=int, default=2000,
                        help="Number of greedy games for behavioral cloning warmup (0 to skip)")
    args = parser.parse_args()

    model = train(args)

    # Eval vs greedy
    print("\nEvaluating vs greedy opponents...")
    for n in [2, 3, 4, 5]:
        if n <= args.num_players + 1:
            wr = evaluate_vs_greedy_batch(model, num_players=n, num_games=1000)
            print(f"  {n} players: win rate = {wr:.1%} (random baseline: {1/n:.1%})")
