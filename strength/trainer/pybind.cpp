#include "st_configuration.h"
#include "st_data_loader.h"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace minizero;
using namespace strength;

std::shared_ptr<Environment> kEnvInstance;

Environment& getEnvInstance()
{
    if (!kEnvInstance) { kEnvInstance = std::make_shared<Environment>(); }
    return *kEnvInstance;
}

PYBIND11_MODULE(strength_py, m)
{
    m.def("load_config_file", [](std::string file_name) {
        env::setUpEnv();
        config::ConfigureLoader cl;
        strength::setConfiguration(cl);
        bool success = cl.loadFromFile(file_name);
        if (config::nn_type_name == "bt" && (config::learner_batch_size != strength::bt_num_batch_size * strength::bt_num_rank_per_batch * strength::bt_num_position_per_rank)) {
            std::cerr << "learner_batch_size (" << config::learner_batch_size
                      << ") should be equal to bt_num_batch_size * bt_num_rank_per_batch * bt_num_position_per_rank ("
                      << strength::bt_num_batch_size << " * "
                      << strength::bt_num_rank_per_batch << " * "
                      << strength::bt_num_position_per_rank << " = "
                      << strength::bt_num_batch_size * strength::bt_num_rank_per_batch * strength::bt_num_position_per_rank << ")" << std::endl;
            success = false;
        }
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("load_config_string", [](std::string conf_str) {
        config::ConfigureLoader cl;
        strength::setConfiguration(cl);
        bool success = cl.loadFromString(conf_str);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("use_gumbel", []() { return config::actor_use_gumbel; });
    m.def("get_zero_replay_buffer", []() { return config::zero_replay_buffer; });
    m.def("use_per", []() { return config::learner_use_per; });
    m.def("get_training_step", []() { return config::learner_training_step; });
    m.def("get_training_display_step", []() { return config::learner_training_display_step; });
    m.def("get_batch_size", []() { return config::learner_batch_size; });
    m.def("get_muzero_unrolling_step", []() { return config::learner_muzero_unrolling_step; });
    m.def("get_n_step_return", []() { return config::learner_n_step_return; });
    m.def("get_learning_rate", []() { return config::learner_learning_rate; });
    m.def("get_momentum", []() { return config::learner_momentum; });
    m.def("get_weight_decay", []() { return config::learner_weight_decay; });
    m.def("get_value_loss_scale", []() { return config::learner_value_loss_scale; });
    m.def("get_game_name", []() { return getEnvInstance().name(); });
    m.def("get_nn_num_input_channels", []() { return getEnvInstance().getNumInputChannels(); });
    m.def("get_nn_input_channel_height", []() { return getEnvInstance().getInputChannelHeight(); });
    m.def("get_nn_input_channel_width", []() { return getEnvInstance().getInputChannelWidth(); });
    m.def("get_nn_num_hidden_channels", []() { return config::nn_num_hidden_channels; });
    m.def("get_nn_hidden_channel_height", []() { return getEnvInstance().getHiddenChannelHeight(); });
    m.def("get_nn_hidden_channel_width", []() { return getEnvInstance().getHiddenChannelWidth(); });
    m.def("get_nn_num_action_feature_channels", []() { return getEnvInstance().getNumActionFeatureChannels(); });
    m.def("get_nn_num_blocks", []() { return config::nn_num_blocks; });
    m.def("get_nn_action_size", []() { return getEnvInstance().getPolicySize(); });
    m.def("get_nn_num_value_hidden_channels", []() { return config::nn_num_value_hidden_channels; });
    m.def("get_nn_discrete_value_size", []() { return kEnvInstance->getDiscreteValueSize(); });
    m.def("get_nn_type_name", []() { return config::nn_type_name; });
    m.def("get_nn_rank_size", []() { return strength::nn_rank_size; });
    m.def("get_bt_num_batch_size", []() { return strength::bt_num_batch_size; });
    m.def("get_bt_num_rank_per_batch", []() { return strength::bt_num_rank_per_batch; });
    m.def("get_bt_num_position_per_rank", []() { return strength::bt_num_position_per_rank; });
    m.def("get_bt_use_weight", []() { return strength::bt_use_weight; });
    m.def("get_training_sgf_dir", []() { return strength::training_sgf_dir; });

    py::class_<StDataLoader>(m, "StDataLoader")
        .def(py::init<std::string>())
        .def("initialize", &StDataLoader::initialize)
        .def("load_data_from_file", &StDataLoader::loadDataFromFile, py::call_guard<py::gil_scoped_release>())
        .def(
            "sample_data", [](StDataLoader& data_loader, py::array_t<float>& features, py::array_t<float>& policy, py::array_t<float>& value) {
                data_loader.getSharedData()->getDataPtr()->features_ = static_cast<float*>(features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->policy_ = static_cast<float*>(policy.request().ptr);
                data_loader.getSharedData()->getDataPtr()->value_ = static_cast<float*>(value.request().ptr);
                data_loader.sampleData();
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "sample_data", [](StDataLoader& data_loader, py::array_t<float>& features, py::array_t<float>& policy, py::array_t<float>& value, py::array_t<float>& rank) {
                data_loader.getSharedData()->getDataPtr()->features_ = static_cast<float*>(features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->policy_ = static_cast<float*>(policy.request().ptr);
                data_loader.getSharedData()->getDataPtr()->value_ = static_cast<float*>(value.request().ptr);
                data_loader.getSharedData()->getDataPtr()->rank_ = static_cast<float*>(rank.request().ptr);
                data_loader.sampleData();
            },
            py::call_guard<py::gil_scoped_release>());
}
