import torch
import torch.nn as nn


class VDNMixer(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, agent_qs):

        """
        agent_qs shape:

        [batch_size, n_agents]

        Example:

        [
            [Q1, Q2, Q3],
            [Q1, Q2, Q3],
            ...
        ]

        VDN simply adds individual Q-values.
        """

        q_total = torch.sum(
            agent_qs,
            dim=1,
            keepdim=True
        )

        return q_total