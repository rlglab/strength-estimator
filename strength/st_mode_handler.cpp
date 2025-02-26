#include "st_mode_handler.h"
#include "evaluator.h"
#include "game_wrapper.h"
#include "git_info.h"
#include "st_actor.h"
#include "st_actor_group.h"
#include "st_configuration.h"
#include "st_console.h"
#include "time_system.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strength {

using namespace minizero;
using namespace minizero::utils;
StModeHandler::StModeHandler()
{
    RegisterFunction("evaluator", this, &StModeHandler::runEvaluator);
    RegisterFunction("mcts_acc", this, &StModeHandler::runMCTSAccuracy);
}
void StModeHandler::loadNetwork(const std::string& nn_file_name, int gpu_id /* = 0 */)
{
    network_ = std::make_shared<StrengthNetwork>();
    network_->loadModel(nn_file_name, gpu_id);
}
void StModeHandler::runConsole()
{
    StConsole console;
    std::string command;
    console.initialize();
    std::cerr << "Successfully started console mode" << std::endl;
    while (getline(std::cin, command)) {
        if (command == "quit") { break; }
        console.executeCommand(command);
    }
}

void StModeHandler::runSelfPlay()
{
    STActorGroup ag;
    ag.run();
}

void StModeHandler::runZeroTrainingName()
{
    std::cout << Environment().name()                  // name for environment
              << "_" << getNetworkAbbeviation()        // network & training algorithm
              << "_" << config::nn_num_blocks << "b"   // number of blocks
              << "x" << config::nn_num_hidden_channels // number of hidden channels
              << "-" << GIT_SHORT_HASH << std::endl;   // git hash info
}

void StModeHandler::runEvaluator()
{
    Evaluator evaluator;
    evaluator.run();
}
void StModeHandler::runMCTSAccuracy()
{
    if (actor_select_action_by_bt) {
        loadNetwork(config::nn_file_name);
        std::string file_name;
        std::map<int, std::vector<std::pair<float, float>>> candidate_Strength;

        std::vector<EnvironmentLoader> env_loaders_cand;

        file_name = strength::candidate_sgf_dir;

        std::cerr << "read: " << file_name << std::endl;
        std::vector<EnvironmentLoader> env_loaders_temp = loadGames(file_name);
        env_loaders_cand.insert(env_loaders_cand.end(), env_loaders_temp.begin(), env_loaders_temp.end());

        candidate_Strength = calculatePosStrength(std::vector<EnvironmentLoader>(env_loaders_cand));

        for (auto weighted_strength : candidate_Strength) {
            for (size_t i = 0; i < weighted_strength.second.size(); i++) {
                strength::cand_strength[i] = weighted_strength.second[i].first / weighted_strength.second[i].second;
                std::cerr << strength::cand_strength[i] << " ";
            }
        }
        std::cerr << std::endl;
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Loading testing sgfs ..." << std::endl;
    std::string file_name = strength::testing_sgf_dir;

    std::cerr << "read: " << file_name << std::endl;
    std::vector<EnvironmentLoader> env_loaders = loadGames(file_name);

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Total loaded " << env_loaders.size() << " games" << std::endl;

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running MCTS accuracy ..." << std::endl;
    STActorGroup ag;
    ag.initialize();
    std::vector<int> game_index(ag.getActors().size(), -1);
    std::vector<std::shared_ptr<actor::BaseActor>>& actors = ag.getActors();
    bool is_done = false;
    int current_game_index = 0;

    std::vector<int> mcts_correct(config::actor_num_simulation, 0);
    std::vector<std::vector<int>> ssa_correct_(temp_for_mcts_ssa_accuracy.size(), std::vector<int>(config::actor_num_simulation, 0));

    std::vector<int> total(config::actor_num_simulation, 0);
    while (!is_done) {
        is_done = true;
        for (size_t i = 0; i < actors.size(); ++i) {
            int move_number = actors[i]->getEnvironment().getActionHistory().size();
            if (game_index[i] != -1 && move_number < static_cast<int>(env_loaders[game_index[i]].getActionPairs().size())) {
                actors[i]->reset();
                is_done = false;
                for (int j = 0; j < move_number; ++j) { actors[i]->act(env_loaders[game_index[i]].getActionPairs()[j].first); }

            } else if (current_game_index < static_cast<int>(env_loaders.size())) {
                is_done = false;
                actors[i]->reset();
                game_index[i] = current_game_index++;
            } else {
                game_index[i] = -1;
                actors[i]->reset();
            }
        }
        if (is_done) { break; }
        ag.step();
        for (size_t i = 0; i < actors.size(); ++i) {
            if (game_index[i] == -1) { continue; }
            std::shared_ptr<StActor> actor = std::static_pointer_cast<StActor>(actors[i]);
            for (size_t j = 0; j < actor->getMCTSActionPerSimulation().size(); ++j) {
                const Action& mcts_action = actor->getMCTSActionPerSimulation()[j];
                const Action& sgf_action = env_loaders[game_index[i]].getActionPairs()[actors[i]->getEnvironment().getActionHistory().size() - 1].first;
                if (mcts_action.getActionID() == sgf_action.getActionID()) { mcts_correct[j]++; }
                for (size_t k = 0; k < temp_for_mcts_ssa_accuracy.size(); k++) {
                    const Action& ssa_action_ = actor->getSSAActionPerSimulation()[k][j];
                    if (ssa_action_.getActionID() == sgf_action.getActionID()) { ssa_correct_[k][j]++; }
                }
                total[j]++;
            }
        }
        // summary
        for (size_t i = 0; i < total.size(); ++i) {
            std::cout << "simulation: " << (i + 1)
                      << ", mcts accuracy: " << (mcts_correct[i] * 100.0 / total[i]) << "% (" << mcts_correct[i] << "/" << total[i] << ")";

            for (size_t j = 0; j < temp_for_mcts_ssa_accuracy.size(); j++) {
                std::cout << ", ssa_" << temp_for_mcts_ssa_accuracy[j] << " accuracy: " << (ssa_correct_[j][i] * 100.0 / total[i]) << "% (" << ssa_correct_[j][i] << "/" << total[i] << ")";
            }
            std::cout << std::endl;
        }
    }
    exit(0);
}
std::map<int, std::vector<std::pair<float, float>>> StModeHandler::calculatePosStrength(const std::vector<EnvironmentLoader>& env_loaders)
{
    if (env_loaders.empty()) { return {}; }
    std::vector<std::pair<float, float>> init(400, {0.0f, 0.0f});
    std::map<int, std::vector<std::pair<float, float>>> results;
    for (size_t i = 0; i < env_loaders.size(); ++i) {
        std::map<int, std::vector<std::pair<float, float>>> tmp = calculatePosStrength(env_loaders[i]);

        for (auto j : tmp) {
            if (results[j.first].size() == 0) results[j.first] = init;
            for (size_t k = 0; k < j.second.size(); k++) {
                results[j.first][k].first += j.second[k].first;
                results[j.first][k].second += j.second[k].second;
            }
        }
    }
    return results;
}
std::map<int, std::vector<std::pair<float, float>>> StModeHandler::calculatePosStrength(const EnvironmentLoader& env_loader)
{
    std::shared_ptr<StrengthNetwork> network = std::static_pointer_cast<StrengthNetwork>(network_);
    int count = 0;
    std::map<int, std::vector<std::pair<float, float>>> results;
    for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
        Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
        std::vector<float> features = calculateFeatures(env_loader, pos, rotation);
        network->pushBack(features);
        count++;
        if ((pos + 1) % 100 == 0 || pos == env_loader.getActionPairs().size() - 1) {
            std::vector<std::shared_ptr<network::NetworkOutput>> output = network->forward();
            for (int pos_ = 0; pos_ < count; ++pos_) {
                std::shared_ptr<StrengthNetworkOutput> s_output = std::static_pointer_cast<StrengthNetworkOutput>(output[pos_]);
                int rank = getRank(env_loader);
                if (strength::bt_use_weight) {
                    results[rank].push_back(std::make_pair(s_output->score_ * s_output->weight_, s_output->weight_));
                } else {
                    results[rank].push_back(std::make_pair(s_output->score_, 1));
                }
            }
            count = 0;
        }
    }
    return results;
}
std::string StModeHandler::getNetworkAbbeviation() const
{
    if (config::nn_type_name == "alphazero") {
        return "az";
    } else if (config::nn_type_name == "bt") {
        return "bt_b" + std::to_string(strength::bt_num_batch_size) + "_r" + std::to_string(strength::bt_num_rank_per_batch) + "_p" + std::to_string(strength::bt_num_position_per_rank);
    } else {
        return config::nn_type_name;
    }
}

} // namespace strength
