# Minesweeper ortaminda step, reset ve reward akisini yonetir.
# Referans ortam davranisindan adapte edilecek ana environment dosyasidir.
import time
from .selenium_controller import SeleniumController
from .template_matcher import TemplateMatcher

class MinesweeperEnv:
    def __init__(self, templates_dir="templates"):
        self.controller = SeleniumController()
        self.matcher = TemplateMatcher(templates_dir=templates_dir)
        self.board_state = None

    def reset(self):
        print("Oyun tahtası sıfırlanıyor...")
        # 'reset.png' veya gülen yüz butonuna tıklatılabilir, şimdilik sayfayı yeniliyoruz
        self.controller.driver.refresh()
        time.sleep(2)
        return self.get_state()

    def get_state(self):
        img_path = self.controller.get_board_screenshot()
        self.board_state = self.matcher.process_board(img_path)
        return self.board_state

    def step(self, action):
        x, y = action
        self.controller.click_cell(x, y)
        time.sleep(0.5)  # Tarayıcının animasyonu oynatması için ufak gecikme
        
        next_state = self.get_state()
        
        # Sprint 1 Baseline için basit kontrol
        # Ekranda "oof.png" (patlamış mayın) eşleşirse done=True olur
        done = any(cell[0] == 'oof' for cell in next_state)
        reward = -1 if done else 0
        
        return next_state, reward, done

    def close(self):
        self.controller.close()