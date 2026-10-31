import random

import numpy as np
import torch
import torch.nn.functional as F

from agents.q_network import QNetwork
from qmix.vdn_mixer import VDNMixer
from qmix.vdn_replay_buffer import VDNReplayBuffer


class VDNAgent:

    def __init__(
        self,
        n_agents=3,
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

        self.n_agents = n_agents

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
        # INDIVIDUAL ONLINE NETWORKS
        # =====================================================

        self.q_networks = []

        for _ in range(self.n_agents):

            network = QNetwork(
                self.state_size,
                self.action_size
            ).to(self.device)

            self.q_networks.append(
                network
            )

        # =====================================================
        # INDIVIDUAL TARGET NETWORKS
        # =====================================================

        self.target_networks = []

        for i in range(self.n_agents):

            network = QNetwork(
                self.state_size,
                self.action_size
            ).to(self.device)

            network.load_state_dict(
                self.q_networks[i].state_dict()
            )

            network.eval()

            self.target_networks.append(
                network
            )

        # =====================================================
        # VDN MIXER
        # =====================================================

        self.mixer = VDNMixer().to(
            self.device
        )

        # =====================================================
        # OPTIMIZER
        # =====================================================

        parameters = []

        for network in self.q_networks:

            parameters.extend(
                network.parameters()
            )

        self.optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate
        )

        # =====================================================
        # REPLAY BUFFER
        # =====================================================

        self.memory = VDNReplayBuffer(
            buffer_size
        )

        # =====================================================
        # TRAINING COUNTER
        # =====================================================

        self.training_steps = 0

    # =========================================================
    # SELECT ACTIONS
    # =========================================================

    def select_actions(
        self,
        observations,
        training=True
    ):

        actions = []

        for i in range(
            self.n_agents
        ):

            # -------------------------------------------------
            # Exploration
            # -------------------------------------------------

            if (
                training
                and
                random.random() < self.epsilon
            ):

                action = random.randrange(
                    self.action_size
                )

            # -------------------------------------------------
            # Exploitation
            # -------------------------------------------------

            else:

                state = torch.tensor(
                    observations[i],
                    dtype=torch.float32
                ).unsqueeze(0).to(
                    self.device
                )

                with torch.no_grad():

                    q_values = self.q_networks[i](
                        state
                    )

                action = int(
                    torch.argmax(
                        q_values,
                        dim=1
                    ).item()
                )

            actions.append(
                action
            )

        return actions

    # =========================================================
    # STORE JOINT TRANSITION
    # =========================================================

    def remember(
        self,
        observations,
        actions,
        reward,
        next_observations,
        done
    ):

        self.memory.push(

            observations,

            actions,

            reward,

            next_observations,

            done

        )

    # =========================================================
    # VDN TRAINING STEP
    # =========================================================

    def train_step(self):

        if len(self.memory) < self.batch_size:

            return None

        # =====================================================
        # SAMPLE JOINT BATCH
        # =====================================================

        (
            observations,
            actions,
            rewards,
            next_observations,
            dones
        ) = self.memory.sample(
            self.batch_size
        )

        observations = torch.tensor(
            observations,
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

        next_observations = torch.tensor(
            next_observations,
            dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        ).to(self.device)

        # =====================================================
        # CURRENT INDIVIDUAL Q VALUES
        # =====================================================

        current_agent_qs = []

        for i in range(
            self.n_agents
        ):

            q_values = self.q_networks[i](
                observations[:, i, :]
            )

            selected_q = q_values.gather(
                1,
                actions[:, i].unsqueeze(1)
            ).squeeze(1)

            current_agent_qs.append(
                selected_q
            )

        # =====================================================
        # STACK INDIVIDUAL Q VALUES
        # =====================================================

        current_agent_qs = torch.stack(
            current_agent_qs,
            dim=1
        )

        # Shape:

        # [batch_size, n_agents]

        # =====================================================
        # VDN MIXING
        # =====================================================

        current_q_total = self.mixer(
            current_agent_qs
        ).squeeze(1)

        # =====================================================
        # TARGET Q VALUES
        # =====================================================

        target_agent_qs = []

        for i in range(
            self.n_agents
        ):

            with torch.no_grad():

                next_q_values = self.target_networks[i](
                    next_observations[:, i, :]
                )

                max_next_q = torch.max(
                    next_q_values,
                    dim=1
                )[0]

            target_agent_qs.append(
                max_next_q
            )

        # =====================================================
        # STACK TARGET Q VALUES
        # =====================================================

        target_agent_qs = torch.stack(
            target_agent_qs,
            dim=1
        )

        # =====================================================
        # MIX TARGET VALUES
        # =====================================================

        target_q_total = self.mixer(
            target_agent_qs
        ).squeeze(1)

        # =====================================================
        # BELLMAN TARGET
        # =====================================================

        target = (

            rewards

            +

            self.gamma
            *
            target_q_total
            *
            (1 - dones)

        )

        # =====================================================
        # LOSS
        # =====================================================

        loss = F.smooth_l1_loss(
            current_q_total,
            target.detach()
        )

        # =====================================================
        # OPTIMIZATION
        # =====================================================

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        # =====================================================
        # UPDATE TARGET NETWORKS
        # =====================================================

        if (
            self.training_steps
            %
            self.target_update
            ==
            0
        ):

            self.update_target_networks()

        return loss.item()

    # =========================================================
    # GET ALL TRAINABLE PARAMETERS
    # =========================================================

    def parameters(self):

        parameters = []

        for network in self.q_networks:

            parameters.extend(
                network.parameters()
            )

        return parameters

    # =========================================================
    # UPDATE TARGET NETWORKS
    # =========================================================

    def update_target_networks(self):

        for i in range(
            self.n_agents
        ):

            self.target_networks[i].load_state_dict(
                self.q_networks[i].state_dict()
            )

    # =========================================================
    # EPSILON DECAY
    # =========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # =========================================================
    # SAVE MODELS
    # =========================================================

    def save(
        self,
        directory="models"
    ):

        import os

        os.makedirs(
            directory,
            exist_ok=True
        )

        for i in range(
            self.n_agents
        ):

            torch.save(

                self.q_networks[i].state_dict(),

                os.path.join(
                    directory,
                    f"vdn_agent_{i + 1}.pth"
                )

            )

    # =========================================================
    # LOAD MODELS
    # =========================================================

    def load(
        self,
        directory="models"
    ):

        import os

        for i in range(
            self.n_agents
        ):

            path = os.path.join(

                directory,

                f"vdn_agent_{i + 1}.pth"

            )

            self.q_networks[i].load_state_dict(

                torch.load(
                    path,
                    map_location=self.device
                )

            )

        self.update_target_networks()