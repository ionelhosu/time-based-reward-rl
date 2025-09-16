import random, collections
import numpy as np

class Replay:
    def __init__(self, capacity=1_000_000):
        self.buf = collections.deque(maxlen=capacity)
    def add(self, *args):
        self.buf.append(tuple(args))
    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s,a,r,s2,d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, np.float32),
                np.stack(s2), np.array(d, np.float32))
    def __len__(self): return len(self.buf)
