#pragma once

#include "mcts.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace strength {

class StMCTSNode : public minizero::actor::MCTSNode {
public:
    StMCTSNode() {}

    void reset() override;
    float getNormalizedPUCTScore(int total_simulation, const std::map<float, int>& tree_value_bound, float& value_s_max, float& value_s_min, float init_q_value = -1.0f);
    std::string toString() const override;

    // setter
    inline void setScore(const float& score) { score_ = score; }
    inline void setWeight(const float& weight) { weight_ = weight; }
    inline void setMoveCount(const int& move_counts) { move_count_ = move_counts; }
    inline void setFirstChild(StMCTSNode* first_child) { minizero::actor::TreeNode::setFirstChild(first_child); }

    // getter
    inline float getScore() const { return score_; }
    inline float getWeight() const { return weight_; }
    inline int getMoveCount() const { return move_count_; }
    inline virtual StMCTSNode* getChild(int index) const override { return (index < num_children_ ? static_cast<StMCTSNode*>(first_child_) + index : nullptr); }

private:
    float score_;
    float weight_;
    int move_count_;
};

class StMCTS : public minizero::actor::MCTS {
public:
    StMCTS(uint64_t tree_node_size)
        : minizero::actor::MCTS(tree_node_size) {}
    void reset() override;
    minizero::actor::MCTSNode* selectChildBySSA(const minizero::actor::MCTSNode* node) const;
    void backup(const std::vector<minizero::actor::MCTSNode*>& node_path, const float value, int move_counts, const float reward = 0.0f, const float score = 0.0f, const float weight = 1.0f);
    std::vector<minizero::actor::MCTSNode*> selectFromNode(minizero::actor::MCTSNode* start_node) override;
    inline StMCTSNode* getRootNode() { return static_cast<StMCTSNode*>(minizero::actor::Tree::getRootNode()); }
    inline const StMCTSNode* getRootNode() const { return static_cast<const StMCTSNode*>(minizero::actor::Tree::getRootNode()); }
    float value_s_max;
    float value_s_min;

protected:
    minizero::actor::TreeNode* createTreeNodes(uint64_t tree_node_size) override { return new StMCTSNode[tree_node_size]; }
    minizero::actor::TreeNode* getNodeIndex(int index) override { return getRootNode() + index; }
    virtual minizero::actor::MCTSNode* selectChildByPUCTScore(const minizero::actor::MCTSNode* node);
};

} // namespace strength
