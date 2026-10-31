import torch
import torch.nn as nn
import torch.optim as optim

from qmix.agent_network import AgentNetwork
from qmix.qmix_mixer import QMixer


class QMIXLearner:

    def __init__(
        self,
        n_agents=3,
        obs_dim=20,
        action_dim=6,
        state_dim=824,
        hidden_dim=64,
        mixing_embed_dim=32,
        lr=0.001,
        gamma=0.99,
        device=None
    ):

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.gamma = gamma

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("QMIX device:", self.device)

        # ==================================================
        # Shared agent network
        # ==================================================

        self.agent_network = AgentNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # Target agent network

        self.target_agent_network = AgentNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.target_agent_network.load_state_dict(
            self.agent_network.state_dict()
        )

        # ==================================================
        # Mixing network
        # ==================================================

        self.mixer = QMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_embed_dim=mixing_embed_dim
        ).to(self.device)

        # Target mixer

        self.target_mixer = QMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_embed_dim=mixing_embed_dim
        ).to(self.device)

        self.target_mixer.load_state_dict(
            self.mixer.state_dict()
        )

        # ==================================================
        # Optimizer
        # ==================================================

        self.optimizer = optim.Adam(
            list(self.agent_network.parameters())
            + list(self.mixer.parameters()),
            lr=lr
        )

        self.loss_function = nn.MSELoss()

    # ======================================================
    # Select actions
    # ======================================================

    def select_actions(
        self,
        observations,
        epsilon=0.1
    ):

        actions = []

        for agent_id in range(self.n_agents):

            obs = torch.tensor(
                observations[agent_id],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            with torch.no_grad():

                q_values = self.agent_network(obs)

            if torch.rand(1).item() < epsilon:

                action = torch.randint(
                    0,
                    self.action_dim,
                    (1,)
                ).item()

            else:

                action = q_values.argmax(
                    dim=1
                ).item()

            actions.append(action)

        return actions

    # ======================================================
    # Calculate training loss
    # ======================================================

    def train_batch(self, batch):

        observations = torch.tensor(
            batch["observations"],
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            batch["actions"],
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            batch["rewards"],
            dtype=torch.float32,
            device=self.device
        )

        next_observations = torch.tensor(
            batch["next_observations"],
            dtype=torch.float32,
            device=self.device
        )

        states = torch.tensor(
            batch["states"],
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            batch["next_states"],
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            batch["dones"],
            dtype=torch.float32,
            device=self.device
        )

        # ==================================================
        # Current Q values
        # ==================================================

        batch_size = observations.shape[0]

        agent_qs = []

        for agent_id in range(self.n_agents):

            obs = observations[:, agent_id, :]

            q_values = self.agent_network(obs)

            selected_q = q_values.gather(
                1,
                actions[:, agent_id].unsqueeze(1)
            ).squeeze(1)

            agent_qs.append(selected_q)

        agent_qs = torch.stack(
            agent_qs,
            dim=1
        )

        # ==================================================
        # Mix individual Q values
        # ==================================================

        q_total = self.mixer(
            agent_qs,
            states
        ).squeeze(1)

        # ==================================================
        # Target Q values
        # ==================================================

        target_agent_qs = []

        with torch.no_grad():

            for agent_id in range(self.n_agents):

                next_obs = next_observations[
                    :, agent_id, :
                ]

                next_q_values = self.target_agent_network(
                    next_obs
                )

                max_next_q = next_q_values.max(
                    dim=1
                )[0]

                target_agent_qs.append(
                    max_next_q
                )

            target_agent_qs = torch.stack(
                target_agent_qs,
                dim=1
            )

            # Mix target values

            target_q_total = self.target_mixer(
                target_agent_qs,
                next_states
            ).squeeze(1)

            target = rewards + (
                self.gamma
                * (1 - dones)
                * target_q_total
            )

        # ==================================================
        # Loss
        # ==================================================

        loss = self.loss_function(
            q_total,
            target
        )

        # ==================================================
        # Backpropagation
        # ==================================================

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.agent_network.parameters())
            + list(self.mixer.parameters()),
            10.0
        )

        self.optimizer.step()

        return loss.item()

    # ======================================================
    # Update target networks
    # ======================================================

    def update_target_networks(self):

        self.target_agent_network.load_state_dict(
            self.agent_network.state_dict()
        )

        self.target_mixer.load_state_dict(
            self.mixer.state_dict()
        )

    # ======================================================
    # Save model
    # ======================================================

    def save(self, path):

        torch.save(
            {
                "agent_network":
                    self.agent_network.state_dict(),

                "mixer":
                    self.mixer.state_dict(),

                "target_agent_network":
                    self.target_agent_network.state_dict(),

                "target_mixer":
                    self.target_mixer.state_dict()
            },
            path
        )

    # ======================================================
    # Load model
    # ======================================================

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.agent_network.load_state_dict(
            checkpoint["agent_network"]
        )

        self.mixer.load_state_dict(
            checkpoint["mixer"]
        )

        self.target_agent_network.load_state_dict(
            checkpoint["target_agent_network"]
        )

        self.target_mixer.load_state_dict(
            checkpoint["target_mixer"]
        )