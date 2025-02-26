#include "st_actor_group.h"
#include "st_actor.h"
#include "st_configuration.h"
#include "strength_network.h"
#include <algorithm>
#include <memory>
#include <torch/cuda.h>

namespace strength {

using namespace minizero;
using namespace minizero::actor;
using namespace minizero::network;

bool STSlaveThread::doCPUJob()
{
    size_t actor_id = getSharedData()->getAvailableActorIndex();
    if (actor_id >= getSharedData()->actors_.size()) { return false; }

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    int network_id = actor_id % getSharedData()->networks_.size();
    int network_output_id = actor->getNNEvaluationBatchIndex();
    if (network_output_id >= 0) {
        assert(network_output_id < static_cast<int>(getSharedData()->network_outputs_[network_id].size()));

        actor->afterNNEvaluation(getSharedData()->network_outputs_[network_id][network_output_id]);

        if (actor->isSearchDone()) { handleSearchDone(actor_id); }
    }
    if (!actor->isSearchDone()) {
        actor->beforeNNEvaluation();
    }
    return true;
}

void STSlaveThread::doGPUJob()
{
    if (id_ >= static_cast<int>(getSharedData()->networks_.size())) { return; }

    std::shared_ptr<StrengthNetwork> network = std::static_pointer_cast<StrengthNetwork>(getSharedData()->networks_[id_]);
    if (network->getBatchSize() > 0) { getSharedData()->network_outputs_[id_] = network->forward(); }
}

void STActorGroup::initialize()
{
    ActorGroup::initialize();
    running_ = true;
}

void STActorGroup::step()
{
    assert(!getSharedData()->actors_.empty());

    std::shared_ptr<BaseActor> actor = getSharedData()->actors_[0];
    size_t move_number = actor->getEnvironment().getActionHistory().size();
    while (true) {
        getSharedData()->actor_index_ = 0;
        for (auto& t : slave_threads_) { t->start(); }
        for (auto& t : slave_threads_) { t->finish(); }
        getSharedData()->do_cpu_job_ = !getSharedData()->do_cpu_job_;
        if (actor->getEnvironment().getActionHistory().size() != move_number) { break; }
    }
}

void STActorGroup::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks);
    getSharedData()->network_outputs_.resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_[gpu_id] = std::make_shared<StrengthNetwork>();
        std::dynamic_pointer_cast<StrengthNetwork>(getSharedData()->networks_[gpu_id])->loadModel(config::nn_file_name, gpu_id);
    }
}

void STActorGroup::createActors()
{
    assert(getSharedData()->networks_.size() > 0);
    std::shared_ptr<Network>& network = getSharedData()->networks_[0];
    uint64_t tree_node_size = static_cast<uint64_t>(config::actor_num_simulation + 1) * network->getActionSize();
    for (int i = 0; i < config::zero_num_parallel_games; ++i) {
        getSharedData()->actors_.emplace_back(std::make_shared<StActor>(tree_node_size));
        getSharedData()->actors_.back()->setNetwork(getSharedData()->networks_[i % getSharedData()->networks_.size()]);
        getSharedData()->actors_.back()->reset();
    }
}

} // namespace strength
