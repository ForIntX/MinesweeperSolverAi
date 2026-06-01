import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from src.agent.dqn_agent import DQNAgent
from src.env.minesweeper_env import MinesweeperEnv


STATE_MAP = {
    "unsolved": -1.0,
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "mine": -5.0,
    "oof": -10.0,
    "gg": 10.0,
    "flag": -2.0,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_state(board_2d):
    flattened = []
    for row in board_2d:
        for cell in row:
            flattened.append(STATE_MAP.get(cell, -1.0))
    return torch.tensor([flattened], dtype=torch.float32)


def get_valid_actions(board_2d):
    valid_actions = []
    cols = len(board_2d[0])

    for row_idx, row in enumerate(board_2d):
        for col_idx, cell in enumerate(row):
            if cell == "unsolved":
                valid_actions.append(row_idx * cols + col_idx)

    return valid_actions


def action_to_coords(action_idx, cols):
    y = action_idx // cols
    x = action_idx % cols
    return x, y


def save_checkpoint(agent, checkpoint_path, rows, cols, episode):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "rows": rows,
            "cols": cols,
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
        },
        checkpoint_path,
    )


def write_metrics(metrics_path, metrics):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["episode", "steps", "reward", "avg_loss", "epsilon", "done"],
        )
        writer.writeheader()
        writer.writerows(metrics)


def run_dqn_training(args):
    print("Sprint 2: DQN egitimi basliyor...")
    set_seed(args.seed)

    rows = args.rows
    cols = args.cols
    total_cells = rows * cols
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = MinesweeperEnv(templates_dir=args.templates_dir, rows=rows, cols=cols)
    agent = DQNAgent(
        input_dim=total_cells,
        output_dim=total_cells,
        device=device,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )

    metrics = []

    try:
        for episode in range(1, args.episodes + 1):
            raw_state = env.reset()
            state = encode_state(raw_state)
            done = False
            steps = 0
            total_reward = 0.0
            losses = []

            while not done and steps < args.max_steps:
                valid_actions = get_valid_actions(raw_state)
                if not valid_actions:
                    done = True
                    break

                action_idx = agent.select_action(state, valid_actions)
                next_raw_state, reward, done = env.step(action_to_coords(action_idx, cols))
                next_state = encode_state(next_raw_state)

                agent.memory.push(state, action_idx, reward, next_state, done)
                loss = agent.optimize_model()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                raw_state = next_raw_state
                total_reward += reward
                steps += 1

            agent.decay_epsilon()
            if episode % args.target_update_freq == 0:
                agent.update_target_net()

            avg_loss = sum(losses) / len(losses) if losses else 0.0
            metrics.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "reward": round(total_reward, 4),
                    "avg_loss": round(avg_loss, 6),
                    "epsilon": round(agent.epsilon, 4),
                    "done": done,
                }
            )

            print(
                "Episode "
                f"{episode}/{args.episodes} | "
                f"Adim: {steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Ort. Loss: {avg_loss:.6f}"
            )

    finally:
        env.close()

    write_metrics(Path(args.metrics_path), metrics)
    save_checkpoint(agent, Path(args.checkpoint_path), rows, cols, args.episodes)
    print(f"Metrikler kaydedildi: {args.metrics_path}")
    print(f"Checkpoint kaydedildi: {args.checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 2 DQN Minesweeper egitimi")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-update-freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--templates-dir", default="templates")
    parser.add_argument("--metrics-path", default="logs/sprint2_metrics.csv")
    parser.add_argument("--checkpoint-path", default="checkpoints/sprint2_dqn.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_dqn_training(parse_args())
