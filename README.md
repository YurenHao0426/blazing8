# Blazing Eights — RL Agent

Self-play PPO agent for the Blazing Eights card game.

## Setup

```bash
pip install torch numpy
```

## Files

- `blazing_env.py` — Game environment (2-5 players)
- `train.py` — PPO self-play training
- `play.py` — Real-time play assistant (input game state, get best move)

## Training

```bash
# Train a 2-player agent (~10min on CPU for 100k episodes)
python train.py --num_players 2 --episodes 100000

# Train for 3 players (may need more episodes)
python train.py --num_players 3 --episodes 200000

# Custom hyperparams
python train.py --num_players 4 --episodes 300000 --lr 1e-4 --ent_coef 0.02
```

Training saves checkpoints every 10k episodes and a final model.

## Real-time Play Assistant

After training, use the assistant during a real game:

```bash
python play.py --model blazing_ppo_final.pt --num_players 3
```

It will prompt you for:
1. Your hand (e.g., `8h,Ks,3d,SWAP`)
2. Top discard card (e.g., `6d`)
3. Active suit if an 8 was played
4. Direction (cw/ccw)
5. Other players' hand sizes
6. Approximate deck size

Then shows ranked action recommendations with probabilities.

## Game Rules

- **56 cards**: standard 52 + 4 Swap cards
- **Match**: suit or rank of top card
- **8**: Wild — choose a suit for next player
- **K**: All other players draw 1
- **Q**: Reverse direction (no effect in 2-player)
- **J**: Skip next player
- **Swap**: Swap your entire hand with next player (playable anytime, no match needed)
- **Can't play**: Draw 1, play it if legal
- **Win**: First to empty hand

## Tips for Better Training

1. **Train per player count** — the optimal policy differs significantly for 2 vs 5 players.
2. **Increase episodes for more players** — larger games have more variance, need more samples.
3. **Opponent modeling** — after self-play, you can fine-tune against specific opponent behaviors by replacing some players with heuristic bots that mimic your friends' tendencies.
4. **Curriculum** — start training with 2 players, then use the trained model to initialize training for 3+ players.
