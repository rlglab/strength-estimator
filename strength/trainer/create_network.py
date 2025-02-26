from minizero.minizero.network.py.alphazero_network import AlphaZeroNetwork
from .rank_network import RankNetwork
from .bt_network import BTNetwork


def create_network(game_name="go",
                   num_input_channels=6,
                   input_channel_height=19,
                   input_channel_width=19,
                   num_hidden_channels=256,
                   hidden_channel_height=19,
                   hidden_channel_width=19,
                   num_blocks=3,
                   action_size=362,
                   num_value_hidden_channels=256,
                   discrete_value_size=1,
                   rank_size=9,
                   network_type_name="alphazero"):

    network = None
    if network_type_name == "alphazero":
        network = AlphaZeroNetwork(game_name,
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
    elif network_type_name == "rank":
        network = RankNetwork(game_name,
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
                              rank_size)
    elif network_type_name == "bt":
        network = BTNetwork(game_name,
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
    else:
        assert False

    return network
