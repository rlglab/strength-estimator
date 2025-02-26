#!/usr/bin/env python

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from strength.trainer.create_network import create_network


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class StDadaLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.StDataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []

        # allocate memory
        if py.get_nn_type_name() == "alphazero":
            self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
        elif py.get_nn_type_name() == "rank":
            self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.rank = np.zeros(py.get_batch_size() * py.get_nn_rank_size(), dtype=np.float32)
        elif py.get_nn_type_name() == "bt":
            self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.rank = np.zeros(py.get_batch_size() * 1, dtype=np.float32)

    def load_data(self):
        for entry in os.listdir(py.get_training_sgf_dir()):
            file_name = os.path.join(py.get_training_sgf_dir(), entry)
            if os.path.isfile(file_name):
                eprint(f"loading data {file_name}")
                self.data_loader.load_data_from_file(file_name)

    def sample_data(self, device='cpu'):
        if py.get_nn_type_name() == "alphazero":
            self.data_loader.sample_data(self.features, self.policy, self.value)
            features = torch.FloatTensor(self.features).view(py.get_batch_size(), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
            policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), py.get_nn_action_size()).to(device)
            value = torch.FloatTensor(self.value).view(py.get_batch_size(), py.get_nn_discrete_value_size()).to(device)
            return features, policy, value
        elif py.get_nn_type_name() == "rank":
            self.data_loader.sample_data(self.features, self.policy, self.value, self.rank)
            features = torch.FloatTensor(self.features).view(py.get_batch_size(), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
            policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), py.get_nn_action_size()).to(device)
            value = torch.FloatTensor(self.value).view(py.get_batch_size(), py.get_nn_discrete_value_size()).to(device)
            rank = torch.FloatTensor(self.rank).view(py.get_batch_size(), py.get_nn_rank_size()).to(device)
            return features, policy, value, rank
        elif py.get_nn_type_name() == "bt":
            self.data_loader.sample_data(self.features, self.policy, self.value, self.rank)
            features = torch.FloatTensor(self.features).view(py.get_batch_size(), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
            policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), py.get_nn_action_size()).to(device)
            value = torch.FloatTensor(self.value).view(py.get_batch_size(), py.get_nn_discrete_value_size()).to(device)
            rank = torch.FloatTensor(self.rank).view(py.get_batch_size(), 1).to(device)
            return features, policy, value, rank


class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

    def load_model(self, training_dir, model_file):
        self.training_step = 0
        self.network = create_network(py.get_game_name(),
                                      py.get_nn_num_input_channels(),
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_rank_size(),
                                      py.get_nn_type_name())
        self.network.to(self.device)
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=py.get_learning_rate(),
                                   momentum=py.get_momentum(),
                                   weight_decay=py.get_weight_decay())
        step_sizes = [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=step_sizes, gamma=0.5)

        if model_file:
            snapshot = torch.load(f"{training_dir}/model/{model_file}", map_location=torch.device('cpu'))
            self.training_step = snapshot['training_step']
            self.network.load_state_dict(snapshot['network'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            self.scheduler.load_state_dict(snapshot['scheduler'])

        # for multi-gpu
        self.network = nn.DataParallel(self.network)

    def save_model(self, training_dir):
        snapshot = {'training_step': self.training_step,
                    'network': self.network.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
        torch.save(snapshot, f"{training_dir}/model/weight_iter_{self.training_step}.pkl")
        torch.jit.script(self.network.module).save(f"{training_dir}/model/weight_iter_{self.training_step}.pt")


def calculate_bt_loss(network_output, label_rank):
    score_reshape = network_output["score"].view(py.get_bt_num_batch_size(), py.get_bt_num_rank_per_batch(), py.get_bt_num_position_per_rank())
    if py.get_bt_use_weight():
        weight_reshape = network_output["weight"].view(py.get_bt_num_batch_size(), py.get_bt_num_rank_per_batch(), py.get_bt_num_position_per_rank())
    else:
        weight_reshape = torch.ones((py.get_bt_num_batch_size(), py.get_bt_num_rank_per_batch(), py.get_bt_num_position_per_rank())).to(network_output["score"].device)
    average_score = (score_reshape * weight_reshape).sum(dim=2) / weight_reshape.sum(dim=2)

    # calculate the ranking loss
    loss_bt = 0
    accuracy_bt = 0
    for i in range(2, py.get_bt_num_rank_per_batch() + 1):
        labels = torch.zeros(py.get_bt_num_batch_size(), i, dtype=average_score.dtype, device=average_score.device)
        labels[:, i - 1] = 1
        loss_bt += -(labels * nn.functional.log_softmax(average_score[:, :i], dim=1)).sum(dim=1).mean()
        # for accuracy
        _, max_output = torch.max(average_score[:, :i], dim=1)
        _, max_label = torch.max(labels, dim=1)
        accuracy_bt += (max_output == max_label).float().mean().item()
    loss_bt /= (py.get_bt_num_rank_per_batch() - 1)
    accuracy_bt /= (py.get_bt_num_rank_per_batch() - 1)
    return loss_bt, accuracy_bt


def add_training_info(training_info, key, value):
    if key not in training_info:
        training_info[key] = 0
    training_info[key] += value


def calculate_accuracy(output, label, batch_size):
    max_output = np.argmax(output.to('cpu').detach().numpy(), axis=1)
    max_label = np.argmax(label.to('cpu').detach().numpy(), axis=1)
    return (max_output == max_label).sum() / batch_size


def train(model, training_dir, data_loader):
    # load data
    data_loader.load_data()

    eprint("start training ...")
    training_info = {}
    for i in range(1, py.get_training_step() + 1):
        model.optimizer.zero_grad()

        if py.get_nn_type_name() == "alphazero":
            features, label_policy, label_value = data_loader.sample_data(model.device)
            network_output = model.network(features)
            loss_policy = -(label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum() / network_output["policy_logit"].shape[0]
            loss_value = torch.nn.functional.mse_loss(network_output["value"], label_value)
            loss = loss_policy + loss_value

            # record training info
            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy, py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
        elif py.get_nn_type_name() == "rank":
            features, label_policy, label_value, label_rank = data_loader.sample_data(model.device)
            network_output = model.network(features)
            loss_policy = -(label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum() / network_output["policy_logit"].shape[0]
            loss_value = torch.nn.functional.mse_loss(network_output["value"], label_value)
            loss_rank = -(label_rank * nn.functional.log_softmax(network_output["rank_logit"], dim=1)).sum() / network_output["rank_logit"].shape[0]
            loss = loss_policy + loss_value + loss_rank

            # record training info
            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy, py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
            add_training_info(training_info, 'loss_rank', loss_rank.item())
            add_training_info(training_info, 'accuracy_rank', calculate_accuracy(network_output["rank_logit"], label_rank, py.get_batch_size()))
        elif py.get_nn_type_name() == "bt":
            features, label_policy, label_value, label_rank = data_loader.sample_data(model.device)
            network_output = model.network(features)
            loss_policy = -(label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum() / network_output["policy_logit"].shape[0]
            loss_value = torch.nn.functional.mse_loss(network_output["value"], label_value)
            loss_bt, accuracy_bt = calculate_bt_loss(network_output, label_rank)
            loss = loss_policy + loss_value + loss_bt

            # record training info
            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy, py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
            add_training_info(training_info, 'loss_bt', loss_bt.item())
            add_training_info(training_info, 'accuracy_bt', accuracy_bt)

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()

        model.training_step += 1
        if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0:
            if model.training_step % (5 * py.get_training_display_step()) == 0:
                model.save_model(training_dir)
            eprint("[{}] nn step {}, lr: {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model.training_step, round(model.optimizer.param_groups[0]["lr"], 6)))
            for loss in training_info:
                eprint("\t{}: {}".format(loss, round(training_info[loss] / py.get_training_display_step(), 5)))
            training_info = {}


if __name__ == '__main__':
    if len(sys.argv) == 5:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        model_file = sys.argv[3]
        conf_file_name = sys.argv[4]

        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['strength_py'], 0)
        py = _temps.strength_py
    else:
        eprint("python train.py game_type training_dir model_file_name conf_file")
        exit(0)

    if py.load_config_file(conf_file_name) is False:
        eprint(f"Failed to load config file {conf_file_name}")
        exit(0)

    data_loader = StDadaLoader(conf_file_name)
    model = Model()

    model_file = model_file.replace('"', '')
    model.load_model(training_dir, model_file)
    if model_file:
        train(model, training_dir, data_loader)
    else:
        model.save_model(training_dir)
