#include "st_mcts.h"
#include "st_configuration.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace strength {

using namespace minizero;
using namespace minizero::actor;

void StMCTSNode::reset()
{
    MCTSNode::reset();
    score_ = 0.0f;
    weight_ = 0.0f;
    move_count_ = 0;
}

float StMCTSNode::getNormalizedPUCTScore(int total_simulation, const std::map<float, int>& tree_value_bound, float& value_s_max, float& value_s_min, float init_q_value /* = -1.0f */)
{
    float puct_bias = config::actor_mcts_puct_init + log((1 + total_simulation + config::actor_mcts_puct_base) / config::actor_mcts_puct_base);
    float value_u = (puct_bias * getPolicy() * sqrt(total_simulation)) / (1 + getCountWithVirtualLoss());
    float value_q = (getCountWithVirtualLoss() == 0 ? init_q_value : getNormalizedMean(tree_value_bound));
    if (actor_select_action_by_bt) {
        float value_s = (getWeight() == 0 ? 0 : fabs(getScore() / getWeight() - cand_strength[getMoveCount()]));
        value_s_max = std::max(value_s_max, value_s);
        if (getWeight() != 0)
            value_s_min = std::min(value_s_min, value_s);
        value_s = (getWeight() == 0 ? 0 : (value_s - value_s_min) / (value_s_max - value_s_min));
        if (getWeight() != 0 && (getScore() / getWeight() - cand_strength[getMoveCount()]) < 0)
            value_s = value_s * 0.5;
        value_s = (puct_bias * value_s * sqrt(total_simulation)) / (1 + getCountWithVirtualLoss());
        return value_u + value_q - s_weight * value_s;
    } else {
        return value_u + value_q;
    }
}

std::string StMCTSNode::toString() const
{
    std::ostringstream oss;
    oss << std::fixed << "p = " << policy_
        << ", p_logit = " << policy_logit_
        << ", p_noise = " << policy_noise_
        << ", v = " << value_
        << ", r = " << reward_
        << ", mean = " << mean_
        << ", score= " << score_
        << ", weight = " << weight_
        << ", strength = " << score_ / weight_
        << ", s = " << score_ / weight_ - cand_strength[move_count_]
        << ", count = " << count_;
    return oss.str();
}
void StMCTS::reset()
{
    MCTS::reset();
    value_s_max = 0.0f;
    value_s_min = 300.0f;
}
MCTSNode* StMCTS::selectChildBySSA(const MCTSNode* node) const
{
    assert(node && !node->isLeaf());

    double sum = 0;
    double softmax_temperature = config::actor_select_action_softmax_temperature;
    MCTSNode* selected = nullptr;
    MCTSNode* best_child = selectChildByMaxCount(node);
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getCount() < best_child->getCount() * 0.1) { continue; }
        if (!selected) { selected = child; }
        double pow_of_count = pow(child->getCount(), softmax_temperature);
        sum += pow_of_count;
    }

    double rand = utils::Random::randReal(sum);
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getCount() < best_child->getCount() * 0.1) { continue; }
        if (rand - pow(child->getCount(), softmax_temperature) < 0) {
            selected = child;
            break;
        }
        rand -= pow(child->getCount(), softmax_temperature);
    }
    assert(selected != nullptr);
    return selected;
}
std::vector<MCTSNode*> StMCTS::selectFromNode(MCTSNode* start_node)
{
    assert(start_node);
    MCTSNode* node = start_node;
    std::vector<MCTSNode*> node_path{node};
    while (!node->isLeaf()) {
        node = selectChildByPUCTScore(node);
        node_path.push_back(node);
    }
    return node_path;
}
MCTSNode* StMCTS::selectChildByPUCTScore(const MCTSNode* node)
{
    assert(node && !node->isLeaf());
    MCTSNode* selected = nullptr;
    int total_simulation = node->getCountWithVirtualLoss() - 1;
    float init_q_value = calculateInitQValue(node);
    float best_score = std::numeric_limits<float>::lowest(), best_policy = std::numeric_limits<float>::lowest();
    for (int i = 0; i < node->getNumChildren(); ++i) {
        StMCTSNode* child = static_cast<StMCTSNode*>(node->getChild(i));
        float score = child->getNormalizedPUCTScore(total_simulation, tree_value_bound_, value_s_max, value_s_min, init_q_value);

        if (score < best_score || (score == best_score && child->getPolicy() <= best_policy)) { continue; }
        best_score = score;
        best_policy = child->getPolicy();
        selected = child;
    }
    assert(selected != nullptr);
    return selected;
}
void StMCTS::backup(const std::vector<MCTSNode*>& node_path, const float value, int move_counts, const float reward /* = 0.0f */, const float score /* = 0.0f */, const float weight /* = 0.0f */)
{
    assert(node_path.size() > 0);
    float updated_value = value;
    node_path.back()->setValue(value);
    node_path.back()->setReward(reward);

    static_cast<StMCTSNode*>(node_path.back())->setScore(score);
    static_cast<StMCTSNode*>(node_path.back())->setWeight(weight);
    static_cast<StMCTSNode*>(node_path.back())->setMoveCount(move_counts);

    for (int i = static_cast<int>(node_path.size() - 1); i >= 0; --i) {
        MCTSNode* node = node_path[i];
        float old_mean = node->getReward() + config::actor_mcts_reward_discount * node->getMean();
        node->add(updated_value);
        if (i != static_cast<int>(node_path.size() - 1)) {
            move_counts--;
            StMCTSNode* st_node = static_cast<StMCTSNode*>(node);
            st_node->setScore(st_node->getScore() + score);
            st_node->setWeight(st_node->getWeight() + weight);
            st_node->setMoveCount(move_counts);
        }
        updateTreeValueBound(old_mean, node->getReward() + config::actor_mcts_reward_discount * node->getMean());
        updated_value = node->getReward() + config::actor_mcts_reward_discount * updated_value;
    }
}

} // namespace strength
