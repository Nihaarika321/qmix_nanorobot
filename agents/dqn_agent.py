import random

import numpy as np
import torch
import torch.nn.functional as F

from agents.q_network import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_size=20,
        action_size=6,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update=100
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.target_update = target_update

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # =====================================================
        # ONLINE NETWORK
        # =====================================================

        self.q_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        # =====================================================
        # TARGET NETWORK
        # =====================================================

        self.target_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        # =====================================================
        # OPTIMIZER
        # =====================================================

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        # =====================================================
        # REPLAY BUFFER
        # =====================================================

        self.memory = ReplayBuffer(
            buffer_size
        )

        self.training_steps = 0

    # =========================================================
    # EPSILON-GREEDY ACTION
    # =========================================================

    def select_action(
        self,
        state,
        training=True
    ):

        # Exploration
        if training and random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            q_values = self.q_network(
                state_tensor
            )

        return int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )

    # =========================================================
    # STORE EXPERIENCE
    # =========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # =========================================================
    # TRAIN
    # =========================================================

    def train_step(self):

        if len(self.memory) < self.batch_size:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            actions,
            dtype=torch.long
        ).to(self.device)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        ).to(self.device)

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        ).to(self.device)

        # =====================================================
        # CURRENT Q VALUES
        # =====================================================

        current_q = self.q_network(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # =====================================================
        # TARGET Q VALUES
        # =====================================================

        with torch.no_grad():

            next_q = self.target_network(
                next_states
            ).max(
                dim=1
            )[0]

            target_q = (
                rewards
                +
                self.gamma
                *
                next_q
                *
                (1 - dones)
            )

        # =====================================================
        # LOSS
        # =====================================================

        loss = F.smooth_l1_loss(
            current_q,
            target_q
        )

        # =====================================================
        # BACKPROPAGATION
        # =====================================================

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        # =====================================================
        # TARGET NETWORK UPDATE
        # =====================================================

        if (
            self.training_steps
            %
            self.target_update
            == 0
        ):

            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return loss.item()

    # =========================================================
    # DECAY EPSILON
    # =========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # =========================================================
    # SAVE MODEL
    # =========================================================

    def save(
        self,
        path
    ):

        torch.save(
            self.q_network.state_dict(),
            path
        )

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load(
        self,
        path
    ):

        self.q_network.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )