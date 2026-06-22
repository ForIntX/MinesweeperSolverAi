"""
Demo Scripti (demo.py)
=======================
Eğitilmiş DQN modelini minesweeper.online üzerinde çalıştırır.

Kullanım:
    python demo.py                              # En iyi modeli kullan
    python demo.py --model checkpoints/best_model.pth
    python demo.py --games 5                    # 5 oyun oyna
    python demo.py --headless                   # Görünmez mod (daha hızlı)

Gereksinimler:
    - pip install selenium webdriver-manager pillow
    - Google Chrome tarayıcısı yüklü olmalı
    - Önce train.py ile model eğitilmiş olmalı
"""

import os
import sys
import time
import argparse
import numpy as np

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.minesweeper_env import MinesweeperEnv
from agent.dqn_agent import DQNAgent
from web.selenium_controller import SeleniumController, GAME_STATUS_WON, GAME_STATUS_LOST

# Tahta boyutu (Beginner modu)
BOARD_WIDTH  = 9
BOARD_HEIGHT = 9
N_CELLS      = BOARD_WIDTH * BOARD_HEIGHT
DEFAULT_MODEL = os.path.join("checkpoints", "best.pth")


def load_agent(model_path: str) -> DQNAgent:
    """Eğitilmiş DQN modelini yükle."""
    if not os.path.exists(model_path):
        print(f"[Demo] HATA: Model dosyası bulunamadı: {model_path}")
        print(f"[Demo] Önce eğitim yapın: python train.py")
        sys.exit(1)

    agent = DQNAgent(state_size=N_CELLS, action_size=N_CELLS)
    agent.load(model_path)
    # Epsilon'u 0 yap: tamamen greedy (öğrenilmiş politika)
    agent.set_epsilon(0.0)
    print(f"[Demo] Model yüklendi: {model_path}")
    print(f"[Demo] Epsilon: {agent.epsilon:.4f} (greedy mod)")
    return agent


def normalize(state: np.ndarray) -> np.ndarray:
    """
    train.py ile AYNI normalizasyon. Ağ eğitim sırasında bu dağılımı
    gördüğü için, demo/inference sırasında da state mutlaka bu fonksiyondan
    geçirilmeli; aksi halde ağ tamamen farklı bir girdi dağılımıyla
    karşılaşır ve anlamlı kararlar veremez.

    -1 (kapalı) → -1.0   (maskeleme işareti)
     0..8 (açık) →  0.0..1.0  (8'e böl)
    """
    out = state.astype(np.float32)
    m = state >= 0
    out[m] = state[m] / 8.0
    return out


def web_board_to_agent_state(web_board: np.ndarray) -> np.ndarray:
    """
    Web'den okunan 2D tahta → Ajan için 1D normalize state vektörü.

    web_board değerleri:
        -1 → kapalı hücre (ajan için -1)
        -2 → bayraklı hücre (ajan için -1 — kapalı gibi)
        -3 → açık mayın
        0-8 → açık sayı

    Parametreler
    ------------
    web_board : (height, width) numpy array

    Dönüş
    ------
    (height*width,) numpy array — train.py'deki normalize() ile aynı ölçekte
    """
    state = web_board.flatten().astype(np.float32)
    # Bayraklı hücreler kapalı gibi muamele görür
    state[state == -2] = -1
    # KRİTİK: ağ eğitimde normalize edilmiş state gördü, burada da aynısı uygulanmalı
    return normalize(state)


def get_valid_actions_from_web_board(web_board: np.ndarray) -> list:
    """
    Web tahtasındaki kapalı ve bayraklı olmayan hücrelerin index listesi.
    DQN yalnızca açılmamış hücrelere tıklamalı (bayraklı hücreler ajan için açılmamış sayılır).
    """
    flat = web_board.flatten()
    return [i for i in range(len(flat)) if flat[i] in [-1, -2]]


def run_demo_game(
    agent: DQNAgent,
    controller: SeleniumController,
    game_num: int,
    delay: float = 0.5,
) -> bool:
    """
    Tek bir demo oyunu çalıştır.

    Parametreler
    ------------
    agent      : Eğitilmiş DQN ajanı
    controller : Selenium kontrol sınıfı
    game_num   : Oyun numarası (log için)
    delay      : Her hamle arasındaki bekleme süresi (saniye)

    Dönüş
    ------
    True → Oyun kazanıldı
    """
    print(f"\n{'='*55}")
    print(f"  OYUN {game_num}")
    print(f"{'='*55}")

    controller.new_game()
    time.sleep(1.5)

    step = 0
    flagged_positions = set()
    steps_with_no_flags = 0
    
    import torch

    while True:
        # Web tahtasını oku
        web_board = controller.read_board_from_dom(
            width=BOARD_WIDTH, height=BOARD_HEIGHT
        )

        # Oyun durumunu kontrol et
        status = controller.get_game_status()
        if status in [GAME_STATUS_WON, GAME_STATUS_LOST]:
            # Oyun bittiğinde doğruluk hesabı yapalım
            final_web_board = controller.read_board_from_dom(width=BOARD_WIDTH, height=BOARD_HEIGHT)
            correct_flags = 0
            for (r, c) in flagged_positions:
                # -2: Doğru bayrak (oyun sonu flag olarak kalır)
                if final_web_board[r, c] == -2:
                    correct_flags += 1
            
            total_flags = len(flagged_positions)
            accuracy = (correct_flags / total_flags * 100) if total_flags > 0 else 0.0
            
            if status == GAME_STATUS_WON:
                print(f"  ★ OYUN KAZANILDI! ({step} adımda)")
            else:
                print(f"  ✗ OYUN KAYBEDİLDİ. ({step} adımda)")
                
            print(f"  [Bayrak Özeti] {total_flags} bayrak kondu, {correct_flags}'i gerçekten mayınmış (%{accuracy:.1f} doğruluk), {steps_with_no_flags} adımda hiç bayrak konmadı (eşik geçilmedi)")
            return status == GAME_STATUS_WON

        # =================================================================
        # DETERMİNİSTİK KURALLAR (Gerçek Mayın Tarlası Mantığı)
        # =================================================================
        flags_placed_this_step = 0
        safe_moves = []
        new_flags = True
        while new_flags:
            new_flags = False
            for r in range(BOARD_HEIGHT):
                for c in range(BOARD_WIDTH):
                    val = web_board[r, c]
                    if 1 <= val <= 8:
                        closed_neighbors = []
                        flagged_count = 0
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH:
                                    n_val = web_board[nr, nc]
                                    if n_val == -1:
                                        closed_neighbors.append((nr, nc))
                                    elif n_val == -2:
                                        flagged_count += 1
                        
                        if len(closed_neighbors) > 0 and len(closed_neighbors) + flagged_count == val:
                            for nr, nc in closed_neighbors:
                                controller.right_click_cell(nr, nc, width=BOARD_WIDTH)
                                flagged_positions.add((nr, nc))
                                web_board[nr, nc] = -2  # Sonraki kontroller için tahtayı güncelle
                                flags_placed_this_step += 1
                                new_flags = True
                                time.sleep(0.05)
                        
                        elif len(closed_neighbors) > 0 and flagged_count == val:
                            for nr, nc in closed_neighbors:
                                if (nr, nc) not in safe_moves:
                                    safe_moves.append((nr, nc))

        if len(safe_moves) > 0:
            for sr, sc in safe_moves:
                controller.click_cell(sr, sc, width=BOARD_WIDTH)
                time.sleep(0.05)
            
            opened_cells = N_CELLS - len([v for v in get_valid_actions_from_web_board(web_board) if web_board.flatten()[v] == -1])
            print(f"  Adım {step+1:3d} | Hamle: Mantıksal Güvenli ({len(safe_moves)} hücre) | Açık: {opened_cells} | Toplam Bayrak: {len(flagged_positions)}")
            step += 1
            time.sleep(delay)
            continue

        if flags_placed_this_step == 0:
            steps_with_no_flags += 1
        # =================================================================

        # Geçerli aksiyonlar (bayraklı hücreleri de geçerli sayıyoruz ki ajan'ın mantığı DOKUNULMAMIŞ olsun,
        # gerçi ajan düşük Q'lu olduğu için seçmeyecektir)
        valid = get_valid_actions_from_web_board(web_board)
        if not valid:
            print(f"  ✗ Geçerli aksiyon kalmadı.")
            return False

        # Ajan state'ini hazırla
        state = web_board_to_agent_state(web_board)

        # Q-değerlerini hesapla ve aksiyonu seç
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor).squeeze().cpu().numpy()

        action = agent.select_action(state, valid)
        row = action // BOARD_WIDTH
        col = action %  BOARD_WIDTH

        opened_cells = N_CELLS - len([v for v in valid if web_board.flatten()[v] == -1])
        print(f"  Adım {step+1:3d} | Hamle: ({row}, {col}) | Açık: {opened_cells} | Toplam Bayrak: {len(flagged_positions)}")

        # Eğer olur da ajan bayraklı bir hücreyi seçerse (çok nadir), önce bayrağı kaldır (toggle)
        if web_board[row, col] == -2:
            controller.right_click_cell(row, col, width=BOARD_WIDTH)
            if (row, col) in flagged_positions:
                flagged_positions.remove((row, col))
            time.sleep(0.05)

        # Tıkla
        controller.click_cell(row, col, width=BOARD_WIDTH)
        time.sleep(delay)

        step += 1

        # Maksimum adım güvenliği
        if step > N_CELLS * 2:
            print(f"  ✗ Maksimum adım aşıldı ({step}).")
            return False


def run_local_demo(agent: DQNAgent, n_games: int = 10):
    """
    Selenium olmadan, yalnızca simülatörde demo çalıştır.
    (Web bağlantısı olmadığında kullanılır)
    """
    print("\n[Demo] Yerel simülatör modu (web bağlantısı yok)")
    env = MinesweeperEnv(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_mines=10)
    wins = 0

    for i in range(1, n_games + 1):
        raw_state = env.reset()
        state = normalize(raw_state)  # KRİTİK: ağ normalize edilmiş state ile eğitildi
        done = False
        steps = 0

        print(f"\n{'='*50}")
        print(f"  OYUN {i}/{n_games}")
        print(f"{'='*50}")

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break

            action = agent.select_action(state, valid)
            row = action // BOARD_WIDTH
            col = action %  BOARD_WIDTH

            raw_state, reward, done, info = env.step(action)
            state = normalize(raw_state)
            steps += 1

            print(f"  Adım {steps:3d} | Hücre ({row},{col}) | "
                  f"Ödül: {reward:+.2f} | Kapalı: {len(valid)-1}")

        result = "KAZANDI ★" if info.get("won") else "KAYBEDİLDİ ✗"
        print(f"  Sonuç: {result} ({steps} adım)")
        env.render()

        if info.get("won"):
            wins += 1

    print(f"\n{'='*50}")
    print(f"  SONUÇ: {wins}/{n_games} oyun kazanıldı ({wins/n_games*100:.1f}%)")
    print(f"{'='*50}")


# ------------------------------------------------------------------
# Ana fonksiyon
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minesweeper DQN Demo")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model dosyası yolu (varsayılan: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--games", type=int, default=5,
        help="Oynayacak oyun sayısı (varsayılan: 5)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Her hamle arasındaki bekleme (saniye, varsayılan: 0.5)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Tarayıcıyı görünmez modda çalıştır",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Web yerine yerel simülatörde demo çalıştır",
    )
    args = parser.parse_args()

    # Ajanı yükle
    agent = load_agent(args.model)

    # Yerel demo modu
    if args.local:
        run_local_demo(agent, n_games=args.games)
        return

    # Web demo modu
    print(f"\n[Demo] minesweeper.online'da {args.games} oyun oynanacak")
    print(f"[Demo] Model: {args.model}")
    print(f"[Demo] Hamle gecikmesi: {args.delay}s")
    print(f"[Demo] Mod: {'Headless' if args.headless else 'Görünür'}")
    print()

    controller = SeleniumController(headless=args.headless)
    wins = 0

    try:
        controller.start()

        for game_num in range(1, args.games + 1):
            won = run_demo_game(agent, controller, game_num, delay=args.delay)
            if won:
                wins += 1
            time.sleep(2.0)  # Oyunlar arası bekleme

        # Sonuç özeti
        print(f"\n{'='*55}")
        print(f"  DEMO TAMAMLANDI")
        print(f"  Kazanılan: {wins}/{args.games} ({wins/args.games*100:.1f}%)")
        print(f"{'='*55}")

    except KeyboardInterrupt:
        print("\n[Demo] Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[Demo] Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nTarayıcıyı kapatmak için Enter'a basın...")
        controller.stop()


if __name__ == "__main__":
    main()
