#include "st_actor.h"
#include "game_wrapper.h"
#include "random.h"
#include "st_configuration.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace strength {

using namespace minizero;
using namespace minizero::actor;
using namespace minizero::utils;

void StActor::beforeNNEvaluation()
{
    mcts_search_data_.node_path_ = selection();
    Environment env_transition = getEnvironmentTransition(mcts_search_data_.node_path_);
    feature_rotation_ = (config::actor_use_random_rotation_features ? static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize)) : Rotation::kRotationNone);
    nn_evaluation_batch_id_ = network_->pushBack(calculateFeatures(env_transition, feature_rotation_));
}

void StActor::afterNNEvaluation(const std::shared_ptr<minizero::network::NetworkOutput>& network_output)
{
    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();

    Environment env_transition = getEnvironmentTransition(node_path);
    std::shared_ptr<StrengthNetworkOutput> output = std::static_pointer_cast<StrengthNetworkOutput>(network_output);
    if (!env_transition.isTerminal()) {
        getMCTS()->expand(leaf_node, calculateActionPolicy(env_transition, output, feature_rotation_));

        int move_counts = env_transition.getActionHistory().size();
        getMCTS()->backup(node_path, output->value_, move_counts, env_transition.getReward(), output->score_, (strength::bt_use_weight ? output->weight_ : 1.0f));
    } else {
        int move_counts = env_transition.getActionHistory().size();
        getMCTS()->backup(node_path, env_transition.getEvalScore(), move_counts, env_transition.getReward(), output->score_, (strength::bt_use_weight ? output->weight_ : 1.0f));
    }

    if (leaf_node == getMCTS()->getRootNode()) { addNoiseToNodeChildren(leaf_node); }
    if (isSearchDone()) { handleSearchDone(); }

    if (config::actor_use_gumbel) { gumbel_zero_.sequentialHalving(getMCTS()); }

    if (leaf_node == getMCTS()->getRootNode()) {
        mcts_action_per_simulation_.clear();

        for (auto& ssa_action_ : ssa_action_per_simulation_) {
            ssa_action_.clear();
        }

    } else {
        mcts_action_per_simulation_.push_back(getMCTS()->selectChildByMaxCount(getMCTS()->getRootNode())->getAction());
        ssa_action_per_simulation_.resize(temp_for_mcts_ssa_accuracy.size());
        for (size_t i = 0; i < temp_for_mcts_ssa_accuracy.size(); i++) {
            config::actor_select_action_softmax_temperature = temp_for_mcts_ssa_accuracy[i];
            ssa_action_per_simulation_[i].push_back(getMCTS()->selectChildBySSA(getMCTS()->getRootNode())->getAction());
        }
    }
}

void StActor::step()
{
    int num_simulation = getMCTS()->getNumSimulation();
    int num_simulation_left = config::actor_num_simulation + 1 - num_simulation;
    int batch_size = std::min(config::actor_mcts_think_batch_size, num_simulation_left);
    assert(batch_size > 0);
    std::vector<std::pair<int, decltype(mcts_search_data_.node_path_)>> node_path_evaluated;
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        beforeNNEvaluation();
        assert(nn_evaluation_batch_id_ == batch_id);
        if (mcts_search_data_.node_path_.back()->getVirtualLoss() == 0) {
            node_path_evaluated.emplace_back(batch_id, std::move(mcts_search_data_.node_path_));
        }
        for (auto node : mcts_search_data_.node_path_) { node->addVirtualLoss(); }
    }
    auto network_output = network_->forward();
    for (auto& evaluation : node_path_evaluated) {
        nn_evaluation_batch_id_ = evaluation.first;
        mcts_search_data_.node_path_ = std::move(evaluation.second);
        afterNNEvaluation(network_output[nn_evaluation_batch_id_]);
        auto virtual_loss = mcts_search_data_.node_path_.back()->getVirtualLoss();
        for (auto node : mcts_search_data_.node_path_) { node->removeVirtualLoss(virtual_loss); }
    }
}

MCTSNode* StActor::decideActionNode()
{
    if (strength::actor_select_action_by_ssa) {
        return getMCTS()->selectChildBySSA(getMCTS()->getRootNode());
    } else {
        return ZeroActor::decideActionNode();
    }
    assert(false);
    return nullptr;
}

std::vector<MCTS::ActionCandidate> StActor::calculateActionPolicy(const Environment& env_transition, const std::shared_ptr<StrengthNetworkOutput>& output, const minizero::utils::Rotation& rotation)
{
    std::vector<MCTS::ActionCandidate> action_candidates;
    for (size_t action_id = 0; action_id < output->policy_.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action)) { continue; }
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        action_candidates.push_back(MCTS::ActionCandidate(action, output->policy_[rotated_id], output->policy_logits_[rotated_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

} // namespace strength
