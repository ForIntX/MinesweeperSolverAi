"""
Selenium Controller (web/selenium_controller.py)
=================================================
Chrome WebDriver ile minesweeper.online'a bağlanır ve:
- Beginner (9x9, 10 mayın) modunu seçer
- Hücrelere koordinat bazlı tıklama yapar
- Oyun durumunu (kazandı/kaybetti/devam) algılar

Gereksinimler:
    pip install selenium webdriver-manager
"""
IPBAN = False
import os
import time
import numpy as np
from typing import Optional, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    WEBDRIVER_MANAGER = True
except ImportError:
    WEBDRIVER_MANAGER = False
    print("[SeleniumController] Uyarı: webdriver-manager kurulu değil. "
          "Chrome driver'ın PATH'de olduğundan emin olun.")

# minesweeper.online'daki oyun durumu sınıfları
GAME_STATUS_PLAYING  = "playing"
GAME_STATUS_WON      = "won"
GAME_STATUS_LOST     = "lost"
GAME_STATUS_UNKNOWN  = "unknown"

if IPBAN == False   :
    TARGET_URL = "https://minesweeper.online/new-game"  # Zorluk seçim (New game) ekranı
else:
    # KRİTİK DÜZELTME: önceden buraya "/home/burak/Masaüstü/..." gibi sabit
    # (hardcoded) bir kişisel yol yazılıydı; bu sadece o bilgisayarda çalışırdı.
    # Artık bu dosyanın bulunduğu konuma göre göreceli (relative) yol hesaplanıyor,
    # böylece proje hangi bilgisayara/klasöre kopyalanırsa kopyalansın çalışır.
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../proje_son_ysa/web
    _PROJECT_ROOT = os.path.dirname(_THIS_DIR)                  # .../proje_son_ysa
    _LOCAL_HTML = os.path.join(_PROJECT_ROOT, "minesweeper_local.html")
    TARGET_URL = "file:///" + _LOCAL_HTML.replace("\\", "/").lstrip("/")

class SeleniumController:
    """
    minesweeper.online için Selenium tabanlı kontrol sınıfı.

    Parametreler
    ------------
    headless    : True → görünmez mod (demo için False önerilir)
    window_size : Tarayıcı boyutu (genişlik, yükseklik)
    """

    def __init__(self, headless: bool = False, window_size: Tuple[int, int] = (1200, 900)):
        self.headless = headless
        self.window_size = window_size
        self.driver: Optional[webdriver.Chrome] = None
        self.board_rect: Optional[dict] = None  # Tahtanın ekrandaki pozisyonu
        self.cell_size: Optional[float] = None  # Piksel cinsinden hücre boyutu

    # ------------------------------------------------------------------
    # Tarayıcı başlatma / kapatma
    # ------------------------------------------------------------------

    def start(self):
        """Chrome tarayıcıyı başlat ve siteye git."""
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")

        options.add_argument(f"--window-size={self.window_size[0]},{self.window_size[1]}")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        # Bot tespitini azalt
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        if WEBDRIVER_MANAGER:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)

        self.driver.get(TARGET_URL)
        self._wait_for_board()
        print(f"[SeleniumController] Tarayıcı başlatıldı: {TARGET_URL}")

    def stop(self):
        """Tarayıcıyı kapat."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("[SeleniumController] Tarayıcı kapatıldı.")

    # ------------------------------------------------------------------
    # Tahta hazırlama
    # ------------------------------------------------------------------

    def _wait_for_board(self, timeout: int = 15):
        """
        Tahta yüklenene kadar bekle.
        minesweeper.online'da tahta #game veya .game-field içinde.
        Önce zorluk seçim ekranını geçmemiz gerekir.
        """
        wait = WebDriverWait(self.driver, timeout)
        
        # 1. Aşama: Zorluk seçimi (Beginner)
        if "minesweeper_local.html" not in TARGET_URL:
            try:
                # Önce "Play for free" varsa ona tıkla (bazı durumlarda çıkabiliyor)
                try:
                    pfb = self.driver.find_elements(By.CSS_SELECTOR, ".btn-play-for-free")
                    if pfb:
                        pfb[0].click()
                        time.sleep(1)
                except Exception:
                    pass

                # 'a.level-select-link[href*="/start/1"]' -> Beginner seçeneği
                beginner_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='/start/1']")))
                beginner_btn.click()
                print("[SeleniumController] Zorluk seçildi: Beginner")
                time.sleep(2)
            except Exception as e:
                print("[SeleniumController] Beginner butonu bulunamadı veya atlandı:", e)

        # 2. Aşama: Oyun tahtasının yüklenmesi
        try:
            # Oyun container'ını bekle
            wait.until(EC.presence_of_element_located((By.ID, "game")))
            time.sleep(1.5)  # JS renderı için küçük buffer
            self._detect_board()
        except TimeoutException:
            print("[SeleniumController] Uyarı: Tahta zaman aşımına uğradı, tekrar deneniyor...")
            time.sleep(2)
            self._detect_board()

    def _detect_board(self):
        """
        Oyun tahtasının koordinatlarını ve hücre boyutunu JavaScript ile tespit et.
        """
        try:
            # minesweeper.online'da her hücre .cell sınıfına sahip
            script = """
            // Engelliyeci reklam ve modalları kaldır
            document.querySelectorAll('.modal, .modal-backdrop, .fc-ab-root, #desktop_sticky_bottom_ad, #bottom_ad, #desktop_sticky_side_ad').forEach(e => e.remove());
            document.body.classList.remove('modal-open');
            
            var cells = document.querySelectorAll('#game .cell');
            if (cells.length === 0) return null;
            var firstCell = cells[0];
            var rect = firstCell.getBoundingClientRect();
            return {
                x: rect.left,
                y: rect.top,
                width: rect.width,
                height: rect.height,
                total_cells: cells.length
            };
            """
            result = self.driver.execute_script(script)
            if result:
                self.cell_size = result["width"]
                self.board_rect = result
                # Sayfayı oyun tahtasının ortasına hizala (tıklamalarda zıplamayı önler)
                self.driver.execute_script("document.getElementById('game').scrollIntoView({block: 'center'});")
                print(f"[SeleniumController] Tahta tespit edildi | "
                      f"Hücre boyutu: {self.cell_size:.1f}px | "
                      f"Toplam hücre: {result['total_cells']}")
            else:
                print("[SeleniumController] Uyarı: Hücre bulunamadı!")
        except Exception as e:
            print(f"[SeleniumController] Tahta tespiti hatası: {e}")

    def new_game(self):
        """Yeni oyun başlat (sayfayı yenile veya 'New Game' düğmesine bas)."""
        try:
            # minesweeper.online'da gülen yüz id'si 'top_area_face'
            new_game_btn = self.driver.find_element(By.ID, "top_area_face")
            
            # Pointer ve Mouse Event simülasyonu
            script = """
            var rect = arguments[0].getBoundingClientRect();
            var cx = rect.left + rect.width / 2;
            var cy = rect.top + rect.height / 2;
            var props = {bubbles:true, cancelable:true, view:window, button: 0, clientX: cx, clientY: cy, pointerId: 1, pointerType: 'mouse'};
            arguments[0].dispatchEvent(new PointerEvent('pointerdown', props));
            arguments[0].dispatchEvent(new MouseEvent('mousedown', props));
            arguments[0].dispatchEvent(new PointerEvent('pointerup', props));
            arguments[0].dispatchEvent(new MouseEvent('mouseup', props));
            arguments[0].dispatchEvent(new MouseEvent('click', props));
            """
            self.driver.execute_script(script, new_game_btn)
            time.sleep(1.0)
            self._detect_board()
            print("[SeleniumController] Yeni oyun başlatıldı.")
        except NoSuchElementException:
            # Bulunamazsa sayfayı yenile
            self.driver.refresh()
            self._wait_for_board()

    # ------------------------------------------------------------------
    # Tıklama
    # ------------------------------------------------------------------

    def click_cell(self, row: int, col: int, width: int = 9, height: int = 9):
        """
        Belirtilen (row, col) koordinatındaki hücreye sol tıkla.

        Parametreler
        ------------
        row, col : 0-tabanlı satır ve sütun indeksi
        width    : Tahta genişliği
        height   : Tahta yüksekliği
        """
        try:
            # Tüm hücreleri al ve index ile seç
            cell_index = row * width + col
            cells = self.driver.find_elements(By.CSS_SELECTOR, "#game .cell")

            if cell_index >= len(cells):
                print(f"[SeleniumController] Hücre bulunamadı: ({row},{col}), index={cell_index}")
                return

            cell = cells[cell_index]
            # minesweeper.online'ın event dinleyicileri clientX ve clientY koordinatlarına ihtiyaç duyar.
            # Koordinatları tam hücrenin ortasından hesaplayıp gönderiyoruz.
            script = """
            var rect = arguments[0].getBoundingClientRect();
            var cx = rect.left + rect.width / 2;
            var cy = rect.top + rect.height / 2;
            var props = {bubbles:true, cancelable:true, view:window, button: 0, clientX: cx, clientY: cy, pointerId: 1, pointerType: 'mouse'};
            arguments[0].dispatchEvent(new PointerEvent('pointerdown', props));
            arguments[0].dispatchEvent(new MouseEvent('mousedown', props));
            arguments[0].dispatchEvent(new PointerEvent('pointerup', props));
            arguments[0].dispatchEvent(new MouseEvent('mouseup', props));
            arguments[0].dispatchEvent(new MouseEvent('click', props));
            """
            self.driver.execute_script(script, cell)
            time.sleep(0.15)

        except Exception as e:
            print(f"[SeleniumController] Tıklama hatası ({row},{col}): {e}")

    def right_click_cell(self, row: int, col: int, width: int = 9):
        """
        Belirtilen hücreye sağ tıkla (bayrak yerleştir/kaldır).
        """
        try:
            cell_index = row * width + col
            cells = self.driver.find_elements(By.CSS_SELECTOR, "#game .cell")
            if cell_index >= len(cells):
                return
            cell = cells[cell_index]
            script = """
            var rect = arguments[0].getBoundingClientRect();
            var cx = rect.left + rect.width / 2;
            var cy = rect.top + rect.height / 2;
            var props = {bubbles:true, cancelable:true, view:window, button: 2, clientX: cx, clientY: cy, pointerId: 1, pointerType: 'mouse'};
            arguments[0].dispatchEvent(new PointerEvent('pointerdown', props));
            arguments[0].dispatchEvent(new MouseEvent('mousedown', props));
            arguments[0].dispatchEvent(new PointerEvent('pointerup', props));
            arguments[0].dispatchEvent(new MouseEvent('mouseup', props));
            arguments[0].dispatchEvent(new MouseEvent('contextmenu', props));
            """
            self.driver.execute_script(script, cell)
            time.sleep(0.15)
        except Exception as e:
            print(f"[SeleniumController] Sağ tıklama hatası ({row},{col}): {e}")

    # ------------------------------------------------------------------
    # Ekran görüntüsü
    # ------------------------------------------------------------------

    def take_screenshot(self, path: str = "screenshot.png") -> str:
        """Ekran görüntüsü al ve kaydet."""
        self.driver.save_screenshot(path)
        return path

    def take_board_screenshot(self, path: str = "board.png") -> str:
        """
        Yalnızca tahta alanının ekran görüntüsünü al.
        PIL/Pillow gerektirir (pip install pillow).
        """
        try:
            from PIL import Image
            import io

            # Tam sayfa screenshot al
            png_bytes = self.driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png_bytes))

            # Tahtayı crop et (JavaScript'ten alınan koordinatlar)
            if self.board_rect and self.cell_size:
                # 9x9 tahta için koordinatlar
                x = int(self.board_rect["x"])
                y = int(self.board_rect["y"])
                w = int(self.cell_size * 9 + 2)
                h = int(self.cell_size * 9 + 2)

                # DPI ölçekleme faktörü
                dpr = self.driver.execute_script("return window.devicePixelRatio;") or 1.0
                x_px = int(x * dpr)
                y_px = int(y * dpr)
                w_px = int(w * dpr)
                h_px = int(h * dpr)

                board_img = img.crop((x_px, y_px, x_px + w_px, y_px + h_px))
                board_img.save(path)
            else:
                img.save(path)

            return path
        except ImportError:
            # PIL yoksa tam screenshot al
            return self.take_screenshot(path)

    # ------------------------------------------------------------------
    # Oyun durumu tespiti
    # ------------------------------------------------------------------

    def get_game_status(self) -> str:
        """
        Oyunun mevcut durumunu tespit et.

        Dönüş
        ------
        'playing' | 'won' | 'lost' | 'unknown'
        """
        try:
            # minesweeper.online'da '#top_area_face' elementi durumu gösterir
            face = self.driver.find_element(By.ID, "top_area_face")
            face_class = face.get_attribute("class") or ""

            if "win" in face_class:
                return GAME_STATUS_WON
            elif "lose" in face_class or "dead" in face_class or "sad" in face_class:
                return GAME_STATUS_LOST
            else:
                return GAME_STATUS_PLAYING

        except NoSuchElementException:
            pass

        # Alternatif: sayfa başlığını kontrol et
        try:
            title = self.driver.title.lower()
            if "win" in title or "congratulation" in title:
                return GAME_STATUS_WON
            if "game over" in title or "lose" in title:
                return GAME_STATUS_LOST
        except Exception:
            pass

        return GAME_STATUS_UNKNOWN

    def is_game_over(self) -> bool:
        """Oyun bitti mi? (kazanıldı veya kaybedildi)"""
        status = self.get_game_status()
        return status in [GAME_STATUS_WON, GAME_STATUS_LOST]

    def get_remaining_mines(self) -> int:
        """Kalan mayın sayacını oku (görüntülenen değer)."""
        try:
            # minesweeper.online'da #mines-count veya benzeri
            counter = self.driver.find_element(By.ID, "mines-count")
            return int(counter.text.strip())
        except Exception:
            return -1

    # ------------------------------------------------------------------
    # Board durumunu okuma (template matching ile değil, DOM'dan)
    # ------------------------------------------------------------------

    def read_board_from_dom(self, width: int = 9, height: int = 9) -> np.ndarray:
        """
        DOM'daki hücre sınıflarından tahta durumunu oku.
        Template matching'e gerek kalmadan hızlı durum okuma.

        Hücre sınıfları (minesweeper.online):
            .cell-closed       → kapalı (-1)
            .cell-flag         → bayrak (-2)
            .cell-mine         → açık mayın (-3)
            .cell-0 .. .cell-8 → açık sayı (0-8)

        Dönüş
        ------
        (height, width) boyutunda numpy array
        """
        board = np.full((height, width), -1, dtype=np.int32)

        try:
            script = """
            var cells = document.querySelectorAll('#game .cell');
            var result = [];
            for (var i = 0; i < cells.length; i++) {
                var cls = cells[i].className;
                var val = -1; // varsayılan kapalı
                if (cls.includes('flag')) {
                    val = -2;
                } else if (cls.includes('mine') || cls.includes('death')) {
                    val = -3;
                } else if (cls.includes('opened') || cls.includes('type')) {
                    var match = cls.match(/type(\\d+)/);
                    if (match) {
                        val = parseInt(match[1]);
                    } else {
                        match = cls.match(/cell-(\\d+)/);
                        if (match) val = parseInt(match[1]);
                        else val = 0;
                    }
                }
                result.push(val);
            }
            return result;
            """
            values = self.driver.execute_script(script)
            if values and len(values) == width * height:
                board = np.array(values, dtype=np.int32).reshape(height, width)
        except Exception as e:
            print(f"[SeleniumController] DOM okuma hatası: {e}")

        return board


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("[Test] SeleniumController — bağlantı testi")
    controller = SeleniumController(headless=False)
    try:
        controller.start()
        print(f"Oyun durumu: {controller.get_game_status()}")
        board = controller.read_board_from_dom()
        print(f"Tahta şekli: {board.shape}")
        print(f"Kapalı hücre sayısı: {(board == -1).sum()}")
        # Orta hücreye tıkla
        controller.click_cell(4, 4)
        time.sleep(1)
        print(f"Tıklama sonrası durum: {controller.get_game_status()}")
        input("Tarayıcıyı kapatmak için Enter'a basın...")
    finally:
        controller.stop()
