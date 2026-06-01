# Experience replay buffer icin gecis kayitlarini saklar ve ornekler.
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Durum, aksiyon, ödül, sonraki durum ve bitiş bilgisini kaydet
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Öğrenme aşamasında rastgele geçmiş hamleleri getir
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)