import random
import numpy as np


class QMIXReplayBuffer:

    def __init__(self, capacity=50000):

        self.capacity = capacity

        self.buffer = []

    def add(
        self,
        observations,
        actions,
        reward,
        next_observations,
        state,
        next_state,
        done
    ):

        experience = (
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            float(reward),
            np.array(next_observations, dtype=np.float32),
            np.array(state, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            float(done)
        )

        if len(self.buffer) >= self.capacity:

            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        observations = np.array(
            [x[0] for x in batch]
        )

        actions = np.array(
            [x[1] for x in batch]
        )

        rewards = np.array(
            [x[2] for x in batch],
            dtype=np.float32
        )

        next_observations = np.array(
            [x[3] for x in batch]
        )

        states = np.array(
            [x[4] for x in batch]
        )

        next_states = np.array(
            [x[5] for x in batch]
        )

        dones = np.array(
            [x[6] for x in batch],
            dtype=np.float32
        )

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "states": states,
            "next_states": next_states,
            "dones": dones
        }

    def __len__(self):

        return len(self.buffer)