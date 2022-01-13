from collections import deque

import numpy as np
import torch


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = zip(
            *[self.memory[idx] for idx in indices]
        )

        return (
            torch.tensor(np.asarray(state_batch)),
            torch.tensor(np.asarray(next_state_batch)),
            torch.tensor(np.asarray(action_batch)),
            torch.tensor(np.asarray(reward_batch)),
            torch.tensor(np.asarray(done_batch)),
        )
