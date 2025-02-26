import torch
import torch.nn as nn
import torch.nn.functional as F
from minizero.minizero.network.py.alphazero_network import AlphaZeroNetwork


class ScoreNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_value_hidden_channels):
        super(ScoreNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(channel_height * channel_width, num_value_hidden_channels)
        self.fc2 = nn.Linear(num_value_hidden_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_height * self.channel_width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class WeightNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_value_hidden_channels):
        super().__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(channel_height * channel_width, num_value_hidden_channels)
        self.fc2 = nn.Linear(num_value_hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_height * self.channel_width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class BTNetwork(AlphaZeroNetwork):
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
                 discrete_value_size):
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
        self.score = ScoreNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, num_value_hidden_channels)
        self.weight = WeightNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, num_value_hidden_channels)

    @torch.jit.export
    def get_type_name(self):
        return "bt"

    def forward(self, state):
        x = self.conv(state)
        x = self.bn(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        # policy
        policy_logit = self.policy(x)
        policy = torch.softmax(policy_logit, dim=1)

        # score
        score = self.score(x)

        # weight
        weight = self.weight(x)

        # value
        if self.discrete_value_size == 1:
            value = self.value(x)
            return {"policy_logit": policy_logit,
                    "policy": policy,
                    "score": score,
                    "weight": weight,
                    "value": value}
        else:
            value_logit = self.value(x)
            value = torch.softmax(value_logit, dim=1)
            return {"policy_logit": policy_logit,
                    "policy": policy,
                    "score": score,
                    "weight": weight,
                    "value_logit": value_logit,
                    "value": value}
