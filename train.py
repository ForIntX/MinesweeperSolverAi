# DQN ajaninin ana egitim dongusunu baslatir ve checkpoint/log uretir.
import random
from src.env.minesweeper_env import MinesweeperEnv
class RandomAgent:
    def __init__(self, rows=16, cols=30):
        self.rows = rows
        self.cols = cols

    def select_action(self):
        # minesweeper.online 0 indekslidir. (0 ile 29 sütun arası, 0 ile 15 satır arası)
        x = random.randint(0, self.cols - 1)
        y = random.randint(0, self.rows - 1)
        return (x, y)

def run_sprint1_baseline():
    print("Sprint 1: Baseline Ajan (Random Agent) Testi Başlıyor...")
    env = MinesweeperEnv(templates_dir="templates")
    agent = RandomAgent()
    
    episodes = 5
    total_steps = 0
    
    try:
        for ep in range(episodes):
            env.reset()
            done = False
            steps = 0
            
            # Sonsuz döngüden kaçınmak için max limit eklendi
            while not done and steps < 100: 
                action = agent.select_action()
                print(f"Episode {ep+1} - Hamle: {action}")
                
                next_state, reward, done = env.step(action)
                steps += 1
            
            total_steps += steps
            print(f"Episode {ep+1} Sonlandı. Atılan adım sayısı: {steps}")
            
    finally:
        env.close()
        
    print(f"\n--- Sonuçlar ---")
    print(f"Ortalama hayatta kalma süresi: {total_steps/episodes} adım.")
    print("Baseline (Rastgele Ajan): Kazanma oranı %0")

if __name__ == "__main__":
    run_sprint1_baseline()