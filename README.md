# Blazing Eights — RL Agent

Self-play PPO agent for the Blazing Eights card game (UNO variant with custom special cards).

## Setup

```bash
pip install torch numpy tqdm
```

## Files

- `blazing_env.py` — Game environment (2-5 players)
- `train.py` — PPO self-play training with greedy warmup
- `versus.py` — Human vs AI interactive game
- `play.py` — Real-time play assistant (input game state, get best move)
- `train_colab.ipynb` — Google Colab GPU training notebook

## Game Rules

**52 cards** (standard deck, Q removed in 2-player) + **4 Swap cards**

| Card | Effect |
|------|--------|
| 8 | Wild — choose a suit for next player |
| K | All other players draw 1 card |
| Q | Reverse direction (removed in 2-player games) |
| J | Skip next player |
| Swap | Swap entire hand with next player (always playable; next card must match the card before the Swap) |

- **Match** top card by suit or rank (8 and Swap are exceptions)
- **Free draw**: you may draw even if you have playable cards
- **After drawing**: play any legal card OR pass (one draw per turn max)
- **Stalemate**: if all players pass without drawing, game ends (fewest cards wins)
- **Win**: first to empty hand
- **Initial hand**: 5 cards each

## Training

```bash
# 2-player (~20min on CPU, 100k episodes)
python train.py --num_players 2 --episodes 100000

# Skip greedy warmup
python train.py --num_players 2 --episodes 100000 --greedy_warmup 0

# Custom hyperparams
python train.py --num_players 4 --episodes 300000 --lr 1e-4 --ent_coef 0.02
```

Training features:
- **Greedy warmup**: behavioral cloning on greedy play before PPO (default 2000 games)
- **CPU/GPU split**: game simulation on CPU, PPO updates on GPU (avoids transfer overhead)
- **CSV log**: `{save_path}_log.csv` with avg_len, loss, vs_greedy win rate every 10k episodes
- Checkpoints every 10k episodes

## Play vs AI

```bash
python versus.py --model blazing_ppo_2p_final.pt
python versus.py --model blazing_ppo_2p_final.pt --num_players 3
python versus.py --model blazing_ppo_2p_final.pt --show_ai  # show AI hand (debug)
```

Controls: number to play card, `d` to draw, `p` to pass, `q` to quit.

## Play Assistant

Input your game state and get ranked action recommendations:

```bash
python play.py --model blazing_ppo_2p_final.pt --num_players 2
```

## Colab GPU Training

Open `train_colab.ipynb` in Google Colab for GPU-accelerated training. See notebook for setup instructions.
