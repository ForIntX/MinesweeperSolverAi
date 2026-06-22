"""
Tekrar Belleği (agent/replay_memory.py)
========================================
DQN için deneyim tekrarı (Experience Replay) tamponu.
Geçiş tuple'larını saklar ve rastgele mini-batch örnekler.
"""

import random
from collections import deque, namedtuple
from typing import List


# Deneyim geçişi için isimli tuple
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    """
    Dairesel (circular) deneyim tekrarı tamponu.

    Parametreler
    ------------
    capacity : Tamponun maksimum kapasitesi (aşılınca en eski silinir)
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        """
        Yeni bir geçiş ekle.

        Parametreler
        ------------
        state      : Mevcut durum (numpy array veya tensor)
        action     : Alınan aksiyon (integer index)
        reward     : Alınan ödül (float)
        next_state : Sonraki durum
        done       : Terminal durum mu? (bool)
        """
        self.memory.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Tampondan rastgele `batch_size` kadar geçiş örnekle.

        Dönüş
        ------
        Transition nesnelerinin listesi
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Mevcut eleman sayısı."""
        return len(self.memory)

    @property
    def is_ready(self) -> bool:
        """Bellek dolu mu? (Eğitim başlatılabilir mi?)"""
        return len(self.memory) >= self.capacity // 10


if __name__ == "__main__":
    # Hızlı test
    import numpy as np

    mem = ReplayMemory(capacity=1000)
    dummy_state = np.zeros(81, dtype=np.float32)

    for i in range(200):
        mem.push(dummy_state, i % 81, 0.3, dummy_state, False)

    print(f"Bellek boyutu: {len(mem)} / {mem.capacity}")
    batch = mem.sample(32)
    print(f"Örneklenen batch boyutu: {len(batch)}")
    print(f"İlk eleman türü: {type(batch[0])}")
    print(f"Örnek aksiyon: {batch[0].action}")
