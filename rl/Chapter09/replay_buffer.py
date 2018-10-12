from numpy.random import choice
import random

from collections import namedtuple, deque


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.replay_buffer = deque(maxlen=buffer_size)  # memory size of replay buffer
        self.batch_size = batch_size  # Training batch size for Neural nets
        self.experience = namedtuple("Experience",
                                     ["state", "action", "reward", "next_state", "done"])  # Experience replay tuple

    def add(self, state, action, reward, next_state, done):
        # Add new experience to replay buffer memory
        experience = self.experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

    def sample(self, batch_size=32):
        # Randomly sample a bacth of experienced tuples from memory

        return random.sample(self.replay_buffer, self.batch_size)

    def __len__(self):
        # Return the current size of replay buffer
        return len(self.replay_buffer)