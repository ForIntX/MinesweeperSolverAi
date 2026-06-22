"""
Eğitim scripti.

Kullanım:
    python train.py                  # 100k episode
    python train.py --episodes 50000
    python train.py --resume

Çıktı:
    checkpoints/best.pth
    checkpoints/last.pth
    logs/training_log.csv
    logs/training_plots.png

Hibrit Strateji:
    Her adımda önce kural tabanlı çözücü devreye girer.
    Çözücü güvenli hücre bulamazsa DQN karar verir.
    Bu sayede ajan gerçekten belirsiz (guess) durumları öğrenir.
"""

import os, sys, csv, time, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.minesweeper_env import MinesweeperEnv
from agent.dqn_agent import DQNAgent

CFG = {
    "width": 9, "height": 9, "n_mines": 10,
    "episodes":     100_000,
    "log_interval": 1_000,
    "lr":           1e-4,
    "gamma":        0.99,
    "eps_start":    1.0,
    "eps_end":      0.05,
    "eps_decay":    40_000,
    "batch":        64,
    "target_update":2_000,
    "memory":       50_000,
    "ckpt_dir":     "checkpoints",
    "log_dir":      "logs",
}


def normalize(state: np.ndarray) -> np.ndarray:
    """
    -1 (kapalı) → -1.0   (maskeleme işareti)
    -2 (bayrak) → -2.0   (kesin mayın)
     0..8 (açık) →  0.0..1.0  (8'e böl)
    """
    out = state.astype(np.float32)
    m = state >= 0
    out[m] = state[m] / 8.0
    return out


def evaluate(env, agent, n=300):
    """Greedy politika ile n oyun oynar, kazanma oranını döndürür."""
    saved = agent.episodes_done
    agent.set_epsilon(0.0)
    wins = 0
    for _ in range(n):
        s, done = env.reset(), False
        while not done:
            safe, flagged = env.rule_based_moves()
            for f in flagged:
                env.board[f] = env.FLAG

            if safe:
                action = safe[0]
            else:
                v = env.valid_actions()
                if not v:
                    break
                s = env.board.copy()
                action = agent.select_action(normalize(s), v)
            s, _, done, info = env.step(action)
        if info.get("won"):
            wins += 1
    agent.episodes_done = saved
    return wins / n


def plot(log_path, out_path):
    eps, wr, rw, ls = [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            eps.append(int(row["ep"]))
            wr.append(float(row["win_rate"]) * 100)
            rw.append(float(row["avg_reward"]))
            ls.append(float(row["avg_loss"]))
    
    if not wr:
        return
        
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(eps, wr,  "#2ecc71"); ax[0].set_title("Win Rate (%)"); ax[0].set_ylim(0, max(max(wr)+5, 10))
    ax[1].plot(eps, rw,  "#3498db"); ax[1].set_title("Avg Reward");   ax[1].axhline(0, color="gray", ls="--")
    ax[2].plot(eps, ls,  "#e74c3c"); ax[2].set_title("Avg Loss")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def train(n_ep, resume=False):
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)
    os.makedirs(CFG["log_dir"],  exist_ok=True)

    env   = MinesweeperEnv(CFG["width"], CFG["height"], CFG["n_mines"])
    agent = DQNAgent(
        state_size      = env.n_cells,
        action_size     = env.n_cells,
        lr              = CFG["lr"],
        gamma           = CFG["gamma"],
        epsilon_start   = CFG["eps_start"],
        epsilon_end     = CFG["eps_end"],
        epsilon_decay   = CFG["eps_decay"],
        batch_size      = CFG["batch"],
        target_update   = CFG["target_update"],
        memory_capacity = CFG["memory"],
    )

    last_ckpt = os.path.join(CFG["ckpt_dir"], "last.pth")
    best_ckpt = os.path.join(CFG["ckpt_dir"], "best.pth")
    if resume and os.path.exists(last_ckpt):
        agent.load(last_ckpt)

    log_path = os.path.join(CFG["log_dir"], "training_log.csv")
    mode = "a" if (resume and os.path.exists(log_path)) else "w"
    csvf = open(log_path, mode, newline="")
    cw   = csv.writer(csvf)
    if mode == "w":
        cw.writerow(["ep", "win_rate", "avg_reward", "avg_loss", "epsilon"])

    best_wr = 0.0
    buf_r, buf_l, buf_w = [], [], 0
    t0 = time.time()

    print(f"\n{'='*55}")
    print(f"  MINESWEEPER DQN  |  {CFG['width']}x{CFG['height']}  |  {CFG['n_mines']} mines")
    print(f"  {n_ep:,} ep  |  device: {agent.device}")
    print(f"  Hibrit: kural-tabanlı + DQN")
    print(f"{'='*55}\n")

    for ep in range(1, n_ep + 1):
        env.reset()
        done  = False
        ep_r  = 0.0

        pending_state = None
        pending_action = None
        pending_reward = 0.0

        while not done:
            safe, flagged = env.rule_based_moves()
            
            # Bulunan bayrakları board üzerine işle (DQN görsün ve bir daha tıklamasın)
            for f in flagged:
                env.board[f] = env.FLAG
                
            if safe:
                for action in safe:
                    if action in env.opened:
                        continue
                    _, r, done, info = env.step(action)
                    ep_r += r
                    
                    # DQN daha önce bir hamle yaptıysa, rule-based'in kazandırdığı ödülü DQN'e atfet
                    if pending_state is not None:
                        pending_reward += r
                        
                    if done:
                        break
                        
                if done and pending_state is not None:
                    agent.remember(pending_state, pending_action, pending_reward, normalize(env.board.copy()), done)
                    pending_state = None
                    
                    loss = agent.learn()
                    if loss is not None:
                        buf_l.append(loss)
            else:
                # Kural tabanlı tıkandı. Eğer beklemede bir DQN transition'ı varsa, ŞU AN tamamlandı.
                if pending_state is not None:
                    agent.remember(pending_state, pending_action, pending_reward, normalize(env.board.copy()), done)
                    pending_state = None
                    
                    loss = agent.learn()
                    if loss is not None:
                        buf_l.append(loss)
                
                v = env.valid_actions()
                if not v:
                    break
                
                # DQN yeni hamlesini yapar
                pending_state = normalize(env.board.copy())
                action = agent.select_action(pending_state, v)
                pending_action = action
                
                _, r, done, info = env.step(action)
                ep_r += r
                pending_reward = r
                
                if done:
                    agent.remember(pending_state, pending_action, pending_reward, normalize(env.board.copy()), done)
                    pending_state = None
                    
                    loss = agent.learn()
                    if loss is not None:
                        buf_l.append(loss)

        agent.episode_done()
        buf_r.append(ep_r)
        if info.get("won"):
            buf_w += 1

        if ep % CFG["log_interval"] == 0:
            wr   = buf_w / CFG["log_interval"]
            avgr = float(np.mean(buf_r))
            avgl = float(np.mean(buf_l)) if buf_l else 0.0
            eps  = agent.epsilon
            spd  = ep / (time.time() - t0)

            print(
                f"ep {ep:7,} | win {wr*100:5.1f}% | "
                f"r {avgr:+6.3f} | loss {avgl:.4f} | ε={eps:.3f} | {spd:.0f}ep/s"
            )
            cw.writerow([ep, f"{wr:.4f}", f"{avgr:.4f}", f"{avgl:.6f}", f"{eps:.4f}"])
            csvf.flush()

            if wr > best_wr:
                best_wr = wr
                agent.save(best_ckpt)
                print(f"  ★ yeni en iyi: {wr*100:.1f}%")

            buf_r, buf_l, buf_w = [], [], 0

        if ep % 10_000 == 0:
            agent.save(last_ckpt)

    csvf.close()
    agent.save(last_ckpt)

    print(f"\n{'='*55}")
    final = evaluate(env, agent, 500)
    print(f"  Final Win Rate (greedy, 500 oyun): {final*100:.1f}%")
    print(f"  En iyi: {best_wr*100:.1f}%  |  Süre: {(time.time()-t0)/60:.0f}dk")
    print(f"{'='*55}\n")

    plot(log_path, os.path.join(CFG["log_dir"], "training_plots.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=CFG["episodes"])
    p.add_argument("--resume",   action="store_true")
    a = p.parse_args()
    train(a.episodes, a.resume)