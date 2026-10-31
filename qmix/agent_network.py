import torch
import torch.nn as nn


class AgentNetwork(nn.Module):

    def __init__(self, obs_dim=20, action_dim=6, hidden_dim=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        """
        obs shape:

        Single:
            [obs_dim]

        Batch:
            [batch, obs_dim]

        Multi-agent batch:
            [batch, agents, obs_dim]
        """

        return self.network(obs)