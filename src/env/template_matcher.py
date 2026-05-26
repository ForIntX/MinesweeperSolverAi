import cv2
import numpy as np
import os

class TemplateMatcher:
    def __init__(self, templates_dir="templates"):
        self.templates = {}
        self.threshold = 0.85  # Sprint 1 raporundaki sabit eşik
        
        # Sadece tahtada aranacak gerçek hücre/ikon şablonlarının isimleri
        valid_templates = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", 
            "mine", "flag", "unsolved", "oof", "reset", "gg"
        ]
        
        for filename in os.listdir(templates_dir):
            if filename.endswith(".png"):
                key = filename.split(".")[0]
                
                # Sadece valid_templates içindeki resimleri belleğe al
                if key in valid_templates:
                    filepath = os.path.join(templates_dir, filename)
                    self.templates[key] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    def process_board(self, image_path):
        main_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        detected_cells = []
        
        for name, template in self.templates.items():
            if template is None:
                continue
            
            result = cv2.matchTemplate(main_img, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.threshold)
            
            # Bulunan noktaları listeye ekle
            for pt in zip(*locations[::-1]):
                detected_cells.append((name, pt))
                
        return detected_cells