import time

from .selenium_controller import SeleniumController
from .template_matcher import TemplateMatcher


class MinesweeperEnv:
    def __init__(self, templates_dir="templates", rows=9, cols=9):
        self.rows = rows
        self.cols = cols
        self.controller = SeleniumController()
        self.matcher = TemplateMatcher(templates_dir=templates_dir, rows=rows, cols=cols)
        self.board_state = None

    def reset(self):
        print("Oyun tahtası sıfırlanıyor...")
        self.controller.reset_game()
        time.sleep(2)
        self.board_state = self.get_state()
        return self.board_state

    def get_state(self):
        dom_state = self.controller.read_board_state(rows=self.rows, cols=self.cols)
        if dom_state is not None:
            self.board_state = dom_state
            return self.board_state

        img_path = self.controller.get_board_screenshot()
        self.board_state = self.matcher.process_board(img_path)
        return self.board_state

    def step(self, action):
        x, y = action
        previous_unsolved = self._count_cells(self.board_state, "unsolved")

        clicked = self.controller.click_cell(x, y)
        time.sleep(0.5)  
        
        next_state = self.get_state()
        next_unsolved = self._count_cells(next_state, "unsolved")

        lost = self._contains_cell(next_state, "oof") or self._contains_cell(next_state, "mine")
        won = next_unsolved == 0 and not lost
        done = lost or won

        if not clicked:
            reward = -1.0
        elif lost:
            reward = -10.0
        elif won:
            reward = 10.0
        elif next_unsolved < previous_unsolved:
            reward = 1.0
        else:
            reward = -0.1
        
        return next_state, reward, done

    def close(self):
        self.controller.close()

    @staticmethod
    def _contains_cell(board_state, cell_name):
        if board_state is None:
            return False
        return any(cell == cell_name for row in board_state for cell in row)

    @staticmethod
    def _count_cells(board_state, cell_name):
        if board_state is None:
            return 0
        return sum(cell == cell_name for row in board_state for cell in row)
