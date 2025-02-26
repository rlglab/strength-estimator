#pragma once

#include "rotation.h"
#include "st_configuration.h"
#include "st_mcts.h"
#include "strength_network.h"
#include "zero_actor.h"
#include <memory>
#include <vector>

namespace strength {

class StActor : public minizero::actor::ZeroActor {
public:
    StActor(uint64_t tree_node_size)
        : minizero::actor::ZeroActor(tree_node_size)
    {
        network_ = nullptr;
    }

    void beforeNNEvaluation() override;
    void afterNNEvaluation(const std::shared_ptr<minizero::network::NetworkOutput>& network_output) override;
    void setNetwork(const std::shared_ptr<minizero::network::Network>& network) override { network_ = std::static_pointer_cast<StrengthNetwork>(network); }
    std::shared_ptr<minizero::actor::Search> createSearch() override { return std::make_shared<StMCTS>(tree_node_size_); }
    std::shared_ptr<StMCTS> getMCTS() { return std::static_pointer_cast<StMCTS>(search_); }
    const std::shared_ptr<StMCTS> getMCTS() const { return std::static_pointer_cast<StMCTS>(search_); }

    inline const std::vector<Action>& getMCTSActionPerSimulation() const { return mcts_action_per_simulation_; }
    inline const std::vector<std::vector<Action>>& getSSAActionPerSimulation() const { return ssa_action_per_simulation_; }

protected:
    void step() override;
    minizero::actor::MCTSNode* decideActionNode() override;
    std::vector<minizero::actor::MCTS::ActionCandidate> calculateActionPolicy(const Environment& env_transition, const std::shared_ptr<StrengthNetworkOutput>& output, const minizero::utils::Rotation& rotation);

    std::shared_ptr<StrengthNetwork> network_;
    std::vector<Action> mcts_action_per_simulation_;
    std::vector<std::vector<Action>> ssa_action_per_simulation_;
};

} // namespace strength
