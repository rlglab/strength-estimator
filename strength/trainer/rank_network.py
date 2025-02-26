import torch
import torch.nn as nn
import torch.nn.functional as F
from minizero.minizero.network.py.network_unit import PolicyNetwork
from minizero.minizero.network.py.alphazero_network import AlphaZeroNetwork


class RankNetwork(AlphaZeroNetwork):
    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 hidden_channel_height,
                 hidden_channel_width,
                 num_blocks,
                 action_size,
                 num_value_hidden_channels,
                 discrete_value_size,
                 rank_size):
        super().__init__(game_name,
                         num_input_channels,
                         input_channel_height,
                         input_channel_width,
                         num_hidden_channels,
                         hidden_channel_height,
                         hidden_channel_width,
                         num_blocks,
                         action_size,
                         num_value_hidden_channels,
                         discrete_value_size)
        self.rank_size = rank_size
        self.rank = PolicyNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, rank_size)

    @torch.jit.export
    def get_type_name(self):
        return "rank"

    @torch.jit.export
    def get_rank_size(self):
        return self.rank_size

    def forward(self, state):
        x = self.conv(state)
        x = self.bn(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        # policy
        policy_logit = self.policy(x)
        policy = torch.softmax(policy_logit, dim=1)

        # rank
        rank_logit = self.rank(x)
        rank = torch.softmax(rank_logit, dim=1)

        # value
        if self.discrete_value_size == 1:
            value = self.value(x)
            return {"policy_logit": policy_logit,
                    "policy": policy,
                    "rank_logit": rank_logit,
                    "rank": rank,
                    "value": value}
        else:
            value_logit = self.value(x)
            value = torch.softmax(value_logit, dim=1)
            return {"policy_logit": policy_logit,
                    "policy": policy,
                    "rank_logit": rank_logit,
                    "rank": rank,
                    "value_logit": value_logit,
                    "value": value}
