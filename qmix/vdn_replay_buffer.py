import random
from collections import deque

import numpy as np


class VDNReplayBuffer:

    def __init__(
        self,
        capacity=50000
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    # =========================================================
    # STORE JOINT EXPERIENCE
    # =========================================================

    def push(
        self,
        observations,
        actions,
        reward,
        next_observations,
        done
    ):

        self.buffer.append(
            (
                np.array(
                    observations,
                    dtype=np.float32
                ),

                np.array(
                    actions,
                    dtype=np.int64
                ),

                float(reward),

                np.array(
                    next_observations,
                    dtype=np.float32
                ),

                float(done)
            )
        )

    # =========================================================
    # SAMPLE BATCH
    # =========================================================

    def sample(
        self,
        batch_size
    ):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        observations = np.array(
            [
                item[0]
                for item in batch
            ],
            dtype=np.float32
        )

        actions = np.array(
            [
                item[1]
                for item in batch
            ],
            dtype=np.int64
        )

        rewards = np.array(
            [
                item[2]
                for item in batch
            ],
            dtype=np.float32
        )

        next_observations = np.array(
            [
                item[3]
                for item in batch
            ],
            dtype=np.float32
        )

        dones = np.array(
            [
                item[4]
                for item in batch
            ],
            dtype=np.float32
        )

        return (
            observations,
            actions,
            rewards,
            next_observations,
            dones
        )

    # =========================================================
    # BUFFER SIZE
    # =========================================================

    def __len__(self):

        return len(self.buffer)