import os
from collections import Counter

import cv2
import numpy as np


class TemplateMatcher:
    CELL_TEMPLATES = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "mine",
        "flag",
        "unsolved",
        "oof",
    }

    def __init__(self, templates_dir="templates", rows=9, cols=9, cell_size=34, threshold=0.85):
        self.templates = {}
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.threshold = threshold

        valid_templates = self.CELL_TEMPLATES | {"reset", "gg"}

        for filename in os.listdir(templates_dir):
            if not filename.endswith(".png"):
                continue

            key = filename.split(".")[0]
            if key not in valid_templates:
                continue

            filepath = os.path.join(templates_dir, filename)
            self.templates[key] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    def process_board(self, image_path):
        main_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if main_img is None:
            raise ValueError(f"Ekran goruntusu okunamadi: {image_path}")

        detections = self._detect_templates(main_img)
        board = [["unsolved" for _ in range(self.cols)] for _ in range(self.rows)]
        if not detections:
            return board

        origin = self._infer_board_origin(detections)
        if origin is None:
            return board

        origin_x, origin_y = origin
        best_by_cell = {}
        tolerance = self.cell_size // 2

        for detection in detections:
            col = round((detection["x"] - origin_x) / self.cell_size)
            row = round((detection["y"] - origin_y) / self.cell_size)
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                continue

            expected_x = origin_x + col * self.cell_size
            expected_y = origin_y + row * self.cell_size
            if abs(detection["x"] - expected_x) > tolerance:
                continue
            if abs(detection["y"] - expected_y) > tolerance:
                continue

            key = (row, col)
            previous = best_by_cell.get(key)
            if previous is None or detection["score"] > previous["score"]:
                best_by_cell[key] = detection

        for (row, col), detection in best_by_cell.items():
            board[row][col] = detection["name"]

        return board

    def _detect_templates(self, main_img):
        detections = []

        for name, template in self.templates.items():
            if name not in self.CELL_TEMPLATES or template is None:
                continue

            result = cv2.matchTemplate(main_img, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.threshold)
            candidates = []

            for point in zip(*locations[::-1]):
                x, y = point
                candidates.append(
                    {
                        "name": name,
                        "x": int(x),
                        "y": int(y),
                        "score": float(result[y, x]),
                    }
                )

            detections.extend(self._suppress_duplicates(candidates))

        return detections

    def _suppress_duplicates(self, candidates):
        kept = []
        min_distance = self.cell_size // 2

        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            too_close = False
            for existing in kept:
                if abs(candidate["x"] - existing["x"]) <= min_distance:
                    if abs(candidate["y"] - existing["y"]) <= min_distance:
                        too_close = True
                        break

            if not too_close:
                kept.append(candidate)

        return kept

    def _infer_board_origin(self, detections):
        candidate_origins = Counter()

        for detection in detections:
            for row in range(self.rows):
                for col in range(self.cols):
                    origin_x = detection["x"] - col * self.cell_size
                    origin_y = detection["y"] - row * self.cell_size
                    candidate_origins[(origin_x, origin_y)] += 1

        if not candidate_origins:
            return None

        return candidate_origins.most_common(1)[0][0]
