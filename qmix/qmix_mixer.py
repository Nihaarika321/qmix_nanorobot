import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):

    def __init__(
        self,
        n_agents=3,
        state_dim=824,
        mixing_embed_dim=32
    ):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # --------------------------------------------------
        # First hypernetwork
        # Generates weights for first mixing layer
        # --------------------------------------------------

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents * mixing_embed_dim)
        )

        # Bias for first layer

        self.hyper_b1 = nn.Linear(
            state_dim,
            mixing_embed_dim
        )

        # --------------------------------------------------
        # Second hypernetwork
        # --------------------------------------------------

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mixing_embed_dim)
        )

        # --------------------------------------------------
        # State-dependent value
        # --------------------------------------------------

        self.V = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, agent_qs, states):

        """
        agent_qs:
            [batch, n_agents]

        states:
            [batch, state_dim]

        returns:
            [batch, 1]
        """

        batch_size = agent_qs.size(0)

        # --------------------------------------------------
        # Generate positive weights
        # This enforces QMIX monotonicity
        # --------------------------------------------------

        w1 = torch.abs(
            self.hyper_w1(states)
        )

        w1 = w1.view(
            batch_size,
            self.n_agents,
            self.embed_dim
        )

        # --------------------------------------------------
        # First mixing layer
        # --------------------------------------------------

        agent_qs = agent_qs.view(
            batch_size,
            1,
            self.n_agents
        )

        hidden = torch.bmm(
            agent_qs,
            w1
        )

        hidden = hidden.view(
            batch_size,
            self.embed_dim
        )

        b1 = self.hyper_b1(states)

        hidden = hidden + b1

        hidden = F.elu(hidden)

        # --------------------------------------------------
        # Second mixing layer
        # --------------------------------------------------

        w2 = torch.abs(
            self.hyper_w2(states)
        )

        w2 = w2.view(
            batch_size,
            self.embed_dim,
            1
        )

        hidden = hidden.view(
            batch_size,
            1,
            self.embed_dim
        )

        y = torch.bmm(
            hidden,
            w2
        )

        y = y.view(
            batch_size,
            1
        )

        # --------------------------------------------------
        # State-dependent bias/value
        # --------------------------------------------------

        v = self.V(states)

        q_total = y + v

        return q_total