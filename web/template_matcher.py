"""
Template Matcher (web/template_matcher.py)
==========================================
OpenCV template matching ile minesweeper.online ekran görüntüsünden
tahta durumunu okur.

Birincil yöntem: selenium_controller.py'deki DOM okuma (daha hızlı ve güvenilir)
İkincil yöntem:  Bu dosyadaki CV2 template matching (yedek/doğrulama için)

Template PNG'leri web/templates/ dizinine yerleştirilmeli:
    0.png, 1.png, ..., 8.png  → Açık hücre sayıları
    closed.png                → Kapalı hücre
    flag.png                  → Bayraklı hücre
    mine.png                  → Patlamış mayın

Template'leri kendiniz kesmek için crop_templates.py kullanın.
"""

import os
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[TemplateMatcher] Uyarı: opencv-python kurulu değil.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Template adı → Durum değeri eşlemesi
TEMPLATE_MAP = {
    "closed": -1,
    "flag":   -2,
    "mine":   -3,
    "0": 0, "1": 1, "2": 2, "3": 3,
    "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
}

MATCH_THRESHOLD = 0.85   # Template eşleşme eşiği


class TemplateMatcher:
    """
    OpenCV template matching ile Minesweeper tahtasını oku.

    Parametreler
    ------------
    templates_dir : Template PNG'lerinin bulunduğu dizin
    width         : Tahta genişliği
    height        : Tahta yüksekliği
    threshold     : Eşleşme eşiği (0-1 arası, yüksek = daha katı)
    """

    def __init__(
        self,
        templates_dir: str = "web/templates",
        width: int = 9,
        height: int = 9,
        threshold: float = MATCH_THRESHOLD,
    ):
        self.templates_dir = templates_dir
        self.width = width
        self.height = height
        self.threshold = threshold
        self.templates: dict = {}   # {durum_değeri: BGR numpy array}
        self.cell_size: int = 0

        if CV2_AVAILABLE:
            self._load_templates()

    # ------------------------------------------------------------------
    # Template yükleme
    # ------------------------------------------------------------------

    def _load_templates(self):
        """templates/ dizinindeki PNG'leri yükle."""
        loaded = 0
        for name, value in TEMPLATE_MAP.items():
            path = os.path.join(self.templates_dir, f"{name}.png")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self.templates[value] = img
                    loaded += 1

        if loaded > 0:
            # Hücre boyutunu ilk template'den al
            first_tmpl = next(iter(self.templates.values()))
            self.cell_size = first_tmpl.shape[0]
            print(f"[TemplateMatcher] {loaded} template yüklendi | "
                  f"Hücre boyutu: {self.cell_size}px")
        else:
            print(f"[TemplateMatcher] Uyarı: {self.templates_dir} içinde template bulunamadı!")
            print("  → crop_templates.py ile template'leri oluşturun.")

    # ------------------------------------------------------------------
    # Template oluşturma yardımcısı
    # ------------------------------------------------------------------

    def save_cell_crop(
        self,
        screenshot_path: str,
        board_x: int,
        board_y: int,
        cell_size: int,
        row: int,
        col: int,
        output_name: str,
    ):
        """
        Screenshot'tan belirli bir hücreyi crop'layıp kaydet.
        Template oluşturmak için kullanılır.

        Parametreler
        ------------
        screenshot_path : Kaynak ekran görüntüsü yolu
        board_x, board_y: Tahtanın sol-üst köşesi (piksel)
        cell_size        : Hücre boyutu (piksel)
        row, col         : Kesip alınacak hücre koordinatı
        output_name      : Çıktı adı (örn: "closed", "1", "flag")
        """
        if not CV2_AVAILABLE:
            print("opencv-python kurulu değil!")
            return

        img = cv2.imread(screenshot_path)
        if img is None:
            print(f"Screenshot okunamadı: {screenshot_path}")
            return

        x = board_x + col * cell_size
        y = board_y + row * cell_size
        cell = img[y:y+cell_size, x:x+cell_size]

        os.makedirs(self.templates_dir, exist_ok=True)
        out_path = os.path.join(self.templates_dir, f"{output_name}.png")
        cv2.imwrite(out_path, cell)
        print(f"Template kaydedildi: {out_path}")

    # ------------------------------------------------------------------
    # Ana eşleştirme fonksiyonu
    # ------------------------------------------------------------------

    def match_cell(self, cell_img: np.ndarray) -> int:
        """
        Tek bir hücre görüntüsünü tüm template'lerle karşılaştır.

        Parametreler
        ------------
        cell_img : BGR formatında hücre görüntüsü (cell_size x cell_size)

        Dönüş
        ------
        Eşleşen durum değeri (-3 → -1 arası veya 0-8)
        -1 (closed) döner eğer eşleşme bulunamazsa
        """
        if not self.templates:
            return -1

        best_val = -1
        best_score = -1.0

        for state_val, tmpl in self.templates.items():
            # Boyutları eşitle
            if tmpl.shape[:2] != cell_img.shape[:2]:
                tmpl_resized = cv2.resize(tmpl, (cell_img.shape[1], cell_img.shape[0]))
            else:
                tmpl_resized = tmpl

            # Normalleştirilmiş çapraz korelasyon
            result = cv2.matchTemplate(cell_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())

            if score > best_score:
                best_score = score
                best_val = state_val

        if best_score < self.threshold:
            return -1   # Eşleşme bulunamadı → kapalı kabul et

        return best_val

    def read_board(
        self,
        screenshot_path: str,
        board_x: int,
        board_y: int,
        cell_size: int,
    ) -> np.ndarray:
        """
        Ekran görüntüsünden tahta durumunu oku.

        Parametreler
        ------------
        screenshot_path      : Kaynak ekran görüntüsü yolu
        board_x, board_y     : Tahtanın sol-üst köşesi (piksel)
        cell_size            : Hücre boyutu (piksel)

        Dönüş
        ------
        (height, width) boyutunda numpy array
        """
        if not CV2_AVAILABLE:
            return np.full((self.height, self.width), -1, dtype=np.int32)

        img = cv2.imread(screenshot_path)
        if img is None:
            print(f"Screenshot okunamadı: {screenshot_path}")
            return np.full((self.height, self.width), -1, dtype=np.int32)

        board = np.zeros((self.height, self.width), dtype=np.int32)

        for r in range(self.height):
            for c in range(self.width):
                x = board_x + c * cell_size
                y = board_y + r * cell_size
                cell_img = img[y:y+cell_size, x:x+cell_size]
                board[r, c] = self.match_cell(cell_img)

        return board


# ------------------------------------------------------------------
# Template yardımcı aracı
# ------------------------------------------------------------------

def create_crop_script(screenshot_path: str, board_x: int, board_y: int, cell_size: int):
    """
    Screenshot'tan tüm template'leri interaktif olarak oluşturmak için
    basit bir yardımcı betik çalıştırır.

    Kullanım:
        python -c "from web.template_matcher import create_crop_script; \
                   create_crop_script('screenshot.png', 80, 200, 30)"
    """
    if not CV2_AVAILABLE:
        print("opencv-python gerekli!")
        return

    print("Template oluşturma rehberi:")
    print("="*50)
    print(f"Screenshot: {screenshot_path}")
    print(f"Tahta başlangıcı: ({board_x}, {board_y})")
    print(f"Hücre boyutu: {cell_size}px")
    print()

    matcher = TemplateMatcher()

    # Kapalı hücre (0,0) — muhtemelen hepsi kapalı başlangıçta
    print("Kapalı hücre template'i oluşturuluyor...")
    matcher.save_cell_crop(screenshot_path, board_x, board_y, cell_size, 0, 0, "closed")

    print("\nDiğer template'ler için oyunu oynayın ve farklı durumlar ortaya çıkınca")
    print("aşağıdaki komutu çalıştırın:")
    print()
    print("  from web.template_matcher import TemplateMatcher")
    print("  m = TemplateMatcher()")
    print("  m.save_cell_crop('screenshot.png', board_x, board_y, cell_size, ROW, COL, 'NAME')")
    print()
    print("NAME değerleri: 0, 1, 2, 3, 4, 5, 6, 7, 8, closed, flag, mine")


if __name__ == "__main__":
    print("[TemplateMatcher] Test modu")
    matcher = TemplateMatcher()
    print(f"Yüklenen template sayısı: {len(matcher.templates)}")
