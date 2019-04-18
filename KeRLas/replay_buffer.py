import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, size, source):
        """Create Replay buffer.
        
        The discipline is:
          * new data appended to the end
          * old data is pushed away from the right
          * sample is chosen randomly

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = deque()
        self._maxsize = size
        self._next_idx = 0
        self._nfed = 0
        self._source = source

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        self._storage.append(data)
        if len(self._storage) > self._maxsize:
            self._storage = self._storage[-self._maxsize:]
        self._nfed += 1

    def addChunk(self, chunk):
        self._storage.extend(chunk)
        if len(self._storage) > self._maxsize:
            self._storage = self._storage[-self._maxsize:]
        self._nfed += len(chunk)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        list of data tuples
        """
        while self._nfed < batch_size:
            chunk, tag = self._source.dataChunk()
            self.addChunk(chunk)
            
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        self._nfed -= batch_size
        return [self._storage[i] for i in idxes]


