import numpy as np
from collections import deque


class MinesweeperEnv:
    """
    9x9 Minesweeper ortamı.
    - İlk hamle her zaman güvenlidir (mayınlar sonradan yerleştirilir).
    - Sıfır hücreler flood-fill ile açılır.
    - State: (81,) int array, -1=kapalı, 0-8=açık komşu sayısı
    """

    CLOSED   = -1
    FLAG     = -2
    MINE_HIT = -3

    def __init__(self, width=9, height=9, n_mines=10):
        self.W = width
        self.H = height
        self.n_mines = n_mines
        self.n_cells = width * height
        self.n_safe  = self.n_cells - n_mines
        self.reset()

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    def _rc(self, idx):
        return divmod(idx, self.W)

    def _idx(self, r, c):
        return r * self.W + c

    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    yield nr, nc

    def _place_mines(self, first_idx):
        fr, fc = self._rc(first_idx)
        safe = {first_idx}
        for nr, nc in self._neighbors(fr, fc):
            safe.add(self._idx(nr, nc))
        pool = [i for i in range(self.n_cells) if i not in safe]
        if len(pool) < self.n_mines:
            pool = [i for i in range(self.n_cells) if i != first_idx]
        chosen = np.random.choice(pool, self.n_mines, replace=False)
        self.mines = set(chosen.tolist())
        self.adj = np.zeros(self.n_cells, dtype=np.int32)
        for m in self.mines:
            r, c = self._rc(m)
            for nr, nc in self._neighbors(r, c):
                self.adj[self._idx(nr, nc)] += 1

    def _flood(self, start):
        q = deque([start])
        while q:
            idx = q.popleft()
            r, c = self._rc(idx)
            for nr, nc in self._neighbors(r, c):
                ni = self._idx(nr, nc)
                if self.board[ni] == self.CLOSED and ni not in self.mines:
                    self.board[ni] = self.adj[ni]
                    self.opened.add(ni)
                    if self.adj[ni] == 0:
                        q.append(ni)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self):
        self.board  = np.full(self.n_cells, self.CLOSED, dtype=np.int32)
        self.mines  = set()
        self.adj    = np.zeros(self.n_cells, dtype=np.int32)
        self.opened = set()
        self.done   = False
        self.first  = True
        return self.board.copy()

    def step(self, action):
        """
        Ödül:
          +1.0  kazanma
          -1.0  mayına basma
           0.0  normal güvenli hamle
          -0.3  zaten açık hücreye tıklama (döngü cezası)
        """
        if self.done:
            return self.board.copy(), 0.0, True, {"won": False}

        if self.first:
            self._place_mines(action)
            self.first = False

        if action in self.opened:
            return self.board.copy(), -0.3, False, {"won": False}

        if action in self.mines:
            self.board[action] = self.MINE_HIT
            self.done = True
            return self.board.copy(), -1.0, True, {"won": False}

        self.board[action] = self.adj[action]
        self.opened.add(action)
        if self.adj[action] == 0:
            self._flood(action)

        if len(self.opened) >= self.n_safe:
            self.done = True
            return self.board.copy(), 1.0, True, {"won": True}

        return self.board.copy(), 0.0, False, {"won": False}

    def valid_actions(self):
        return [i for i in range(self.n_cells) if self.board[i] == self.CLOSED]

    # eski isim
    def get_valid_actions(self):
        return self.valid_actions()

    def render(self):
        """
        Tahtanın metin tabanlı (terminal) görünümünü basar.
        demo.py --local modunda her oyun sonunda çağrılır.
        Sembol anahtarı: . = kapalı, F = bayrak, * = patlamış mayın, 0-8 = açık sayı
        """
        symbols = {
            self.CLOSED:   ".",
            self.FLAG:     "F",
            self.MINE_HIT: "*",
        }
        lines = []
        for r in range(self.H):
            row_cells = []
            for c in range(self.W):
                v = int(self.board[self._idx(r, c)])
                row_cells.append(symbols.get(v, str(v)))
            lines.append(" ".join(row_cells))
        print("\n".join(lines))

    # ── Kural tabanlı çözücü ─────────────────────────────────────────────────

    def rule_based_moves(self):
        """
        Açık sayı hücrelerine bakarak kesin güvenli ve kesin mayın
        hücrelerini döndürür.

        Döndürür
        --------
        safe  : kesin açılabilir hücre indeksleri listesi
        mines : kesin mayın hücre indeksleri listesi
        """
        flagged = set()   # kesin mayın
        safe    = set()   # kesin güvenli

        for idx in range(self.n_cells):
            if self.board[idx] < 0:
                continue                        # kapalı ya da mayın-hit
            number = int(self.board[idx])
            if number == 0:
                continue

            r, c = self._rc(idx)
            closed_nbrs  = []
            flagged_nbrs = 0

            for nr, nc in self._neighbors(r, c):
                ni = self._idx(nr, nc)
                if self.board[ni] == self.CLOSED:
                    if ni in flagged:
                        flagged_nbrs += 1
                    else:
                        closed_nbrs.append(ni)
                elif self.board[ni] == self.FLAG:
                    flagged_nbrs += 1

            remaining = number - flagged_nbrs

            # Tüm kapalı komşular mayın → hepsini işaretle
            if remaining == len(closed_nbrs) and remaining > 0:
                for ni in closed_nbrs:
                    flagged.add(ni)

            # Kalan mayın sayısı 0 → tüm kapalı komşular güvenli
            elif remaining == 0:
                for ni in closed_nbrs:
                    safe.add(ni)

        # Flaglenenler güncellendikten sonra ikinci geçiş (Board.py update_from benzeri)
        changed = True
        while changed:
            changed = False
            for idx in range(self.n_cells):
                if self.board[idx] <= 0:
                    continue
                number = int(self.board[idx])
                r, c = self._rc(idx)
                closed_nbrs, flagged_nbrs = [], 0
                for nr, nc in self._neighbors(r, c):
                    ni = self._idx(nr, nc)
                    if self.board[ni] == self.CLOSED:
                        if ni in flagged:
                            flagged_nbrs += 1
                        else:
                            closed_nbrs.append(ni)
                    elif self.board[ni] == self.FLAG:
                        flagged_nbrs += 1
                remaining = number - flagged_nbrs
                if remaining == len(closed_nbrs) and remaining > 0:
                    before = len(flagged)
                    for ni in closed_nbrs:
                        flagged.add(ni)
                    if len(flagged) > before:
                        changed = True
                elif remaining == 0:
                    before = len(safe)
                    for ni in closed_nbrs:
                        safe.add(ni)
                    if len(safe) > before:
                        changed = True

        # Güvenli listeden zaten açıkları ve mayınları çıkar
        safe = safe - flagged - self.opened
        flagged = flagged - self.opened

        return list(safe), list(flagged)