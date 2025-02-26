#include "evaluator.h"
#include "game_wrapper.h"
#include "random.h"
#include "st_configuration.h"
#include "strength_network.h"
#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <time_system.h>
#include <torch/cuda.h>
#include <utility>
#include <vector>

namespace strength {

using namespace minizero;
using namespace minizero::network;
using namespace minizero::utils;

int EvaluatorSharedData::getNextSgfIndex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return (sgf_index_ < static_cast<int>(sgfs_.size()) ? sgf_index_++ : sgfs_.size());
}

void EvaluatorSlaveThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void EvaluatorSlaveThread::runJob()
{
    while (true) {
        size_t sgf_index = getSharedData()->getNextSgfIndex();
        if (sgf_index >= getSharedData()->sgfs_.size()) { break; }

        // load game and calculate features
        const std::string& sgf = getSharedData()->sgfs_[sgf_index];
        GameData game_data;
        game_data.env_loader_ = loadGame(sgf);
        if (game_data.env_loader_.getActionPairs().empty()) { continue; }
        std::vector<Rotation> rotations;
        std::vector<std::vector<float>> features;
        for (size_t pos = 0; pos < game_data.env_loader_.getActionPairs().size(); ++pos) {
            Rotation rotation = (config::actor_use_random_rotation_features ? static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize)) : Rotation::kRotationNone);
            rotations.push_back(rotation);
            features.push_back(calculateFeatures(game_data.env_loader_, pos, rotation));
        }

        // forward
        std::vector<std::shared_ptr<NetworkOutput>> network_outputs;
        int network_id = id_ % static_cast<int>(getSharedData()->networks_.size());
        {
            std::lock_guard<std::mutex> lock(*getSharedData()->network_mutexes_[network_id]);
            std::shared_ptr<StrengthNetwork> network = std::static_pointer_cast<StrengthNetwork>(getSharedData()->networks_[network_id]);
            for (auto& f : features) { network->pushBack(f); }
            network_outputs = network->forward();
        }

        // save results
        for (size_t pos = 0; pos < game_data.env_loader_.getActionPairs().size(); ++pos) {
            std::shared_ptr<StrengthNetworkOutput> network_output = std::static_pointer_cast<StrengthNetworkOutput>(network_outputs[pos]);
            if (config::nn_type_name == "rank") {
                game_data.rank_.push_back(network_output->rank_);
            } else if (config::nn_type_name == "bt") {
                game_data.scores_.push_back(network_output->score_);
                game_data.weights_.push_back(network_output->weight_);
            }
            game_data.value_.push_back(network_output->value_);
            std::vector<float> policy;
            for (size_t action_id = 0; action_id < network_output->policy_.size(); ++action_id) {
                int rotated_id = game_data.env_loader_.getRotateAction(action_id, rotations[pos]);
                policy.push_back(network_output->policy_[rotated_id]);
            }
            game_data.policy_.push_back(policy);
        }

        std::lock_guard<std::mutex> lock(getSharedData()->mutex_);
        getSharedData()->env_loaders_map_[getRank(game_data.env_loader_)].push_back(game_data);
        if (sgf_index > 0 && (sgf_index % static_cast<int>(getSharedData()->sgfs_.size() * 0.1)) == 0) { std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << sgf_index << " / " << getSharedData()->sgfs_.size() << std::endl; }
    }
}

void Evaluator::run()
{
    initialize();

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Evaluator mode: " << strength::evaluator_mode << std::endl;
    if (strength::evaluator_mode == "game_prediction") {
        runGamePrediction();
    } else if (strength::evaluator_mode == "move_prediction") {
        runMovePrediction();
    } else {
        std::cerr << "Unknown evaluator mode: " << strength::evaluator_mode << std::endl;
    }
}

void Evaluator::initialize()
{
    createSlaveThreads(config::learner_num_thread);
    createNeuralNetworks();
}

void Evaluator::createNeuralNetworks()
{
    int num_networks = static_cast<int>(torch::cuda::device_count());
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_[gpu_id] = std::make_shared<StrengthNetwork>();
        getSharedData()->network_mutexes_.emplace_back(new std::mutex);
        std::static_pointer_cast<StrengthNetwork>(getSharedData()->networks_[gpu_id])->loadModel(config::nn_file_name, gpu_id);
    }
}

void Evaluator::runGamePrediction()
{
    if (config::nn_type_name == "bt") {
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running candidate sgfs ..." << std::endl;
        evaluateGames(strength::candidate_sgf_dir);
        std::map<int, std::vector<GameData>> candidate = getSharedData()->env_loaders_map_;

        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Calculating candidate strength ... " << std::endl;
        std::vector<std::pair<int, float>> rank_scores;
        for (auto& m : candidate) {
            int num_positions = 0;
            std::pair<int, float> rank_score = {m.first, 0.0f};
            for (auto& game_data : m.second) {
                int temp = 0;
                int lower_bound = 0;
                int upper_bound = 1000;

                if (strength::select_move == "last_50_moves") {
                    lower_bound = static_cast<int>(game_data.scores_.size()) - 50;
                }
                if (strength::select_move == "first_50_moves") {
                    upper_bound = 50;
                }

                for (auto& score : game_data.scores_) {
                    if (temp >= lower_bound) {
                        rank_score.second += score;
                        num_positions++;
                    }
                    temp++;
                    if (temp >= upper_bound) break;
                }
            }
            rank_score.second /= num_positions;
            rank_scores.push_back(rank_score);
        }
        std::sort(rank_scores.begin(), rank_scores.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Candidating strength: " << std::endl;
        for (auto& rank_score : rank_scores) { std::cerr << "\t" << rank_score.first << " " << rank_score.second << std::endl; }

        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running testing sgfs ..." << std::endl;
        evaluateGames(strength::testing_sgf_dir);
        std::map<int, std::vector<GameData>> testing = getSharedData()->env_loaders_map_;

        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing ranks ..." << std::endl;
        summarizeGamePrediction(rank_scores, testing);
    } else if (config::nn_type_name == "rank") {
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running testing sgfs ..." << std::endl;
        evaluateGames(strength::testing_sgf_dir);
        std::map<int, std::vector<GameData>> testing = getSharedData()->env_loaders_map_;
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing ranks ..." << std::endl;
        summarizeGamePrediction(testing);
    }
}

void Evaluator::runMovePrediction()
{
    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running testing sgfs ..." << std::endl;
    evaluateGames(strength::testing_sgf_dir);

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Calculating rank orders ... " << std::endl;
    int max_order = 0;
    std::map<int, std::vector<int>> rank_orders;
    for (auto& m : getSharedData()->env_loaders_map_) {
        for (auto& game_data : m.second) {
            const EnvironmentLoader& env_loader = game_data.env_loader_;
            for (size_t i = 0; i < env_loader.getActionPairs().size(); ++i) {
                const Action& action = env_loader.getActionPairs()[i].first;
                float policy = game_data.policy_[i][action.getActionID()];
                int policy_order = 1;
                for (size_t j = 0; j < game_data.policy_[i].size(); ++j) { policy_order += (game_data.policy_[i][j] > policy); }
                max_order = std::max(max_order, policy_order);

                if (policy_order >= static_cast<int>(rank_orders[m.first].size())) { rank_orders[m.first].resize(policy_order + 1, 0); }
                ++rank_orders[m.first][policy_order];
            }
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing accuracy ... " << std::endl;
    int all_total = 0;
    std::map<int, std::pair<int, int>> accumulated_accuracy;
    for (auto& m : rank_orders) {
        std::cout << "\t" << m.first; // header
        accumulated_accuracy[m.first] = {0, std::accumulate(rank_orders[m.first].begin(), rank_orders[m.first].end(), 0)};
        all_total += accumulated_accuracy[m.first].second;
    }
    std::cout << "\tall" << std::endl;
    for (int i = 1; i <= max_order; ++i) {
        std::cout << i;
        int all_correct = 0;
        for (auto& m : rank_orders) {
            if (i < static_cast<int>(rank_orders[m.first].size())) { accumulated_accuracy[m.first].first += rank_orders[m.first][i]; }
            all_correct += accumulated_accuracy[m.first].first;
            std::cout << "\t" << accumulated_accuracy[m.first].first * 1.0f / accumulated_accuracy[m.first].second;
        }
        std::cout << "\t" << all_correct * 1.0f / all_total << std::endl;
    }
}

void Evaluator::evaluateGames(const std::string& directory)
{
    getSharedData()->sgfs_.clear();
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::string sgf;
        std::ifstream file(entry.path());
        while (std::getline(file, sgf)) { getSharedData()->sgfs_.push_back(sgf); }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Successfully loaded " << getSharedData()->sgfs_.size() << " sgfs" << std::endl;
    getSharedData()->sgf_index_ = 0;
    getSharedData()->env_loaders_map_.clear();
    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
}
void Evaluator::summarizeGamePrediction(const std::map<int, std::vector<GameData>>& testing)
{
    const int max_games = 100;
    const int repeat_times = 500;
    std::map<int, std::vector<float>> rank_accuracy;
    for (auto& rank_score : testing) {
        int rank = rank_score.first;
        rank_accuracy[rank].resize(max_games, 0.0f);
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Predicting rank " << rank << " ... " << std::endl;
        for (int used_games = 1; used_games <= max_games; ++used_games) {
            int correct = 0;
            int correct1 = 0;
            int correct_1 = 0;
            for (int i = 0; i < repeat_times; ++i) {
                std::vector<float> sum_rank(strength::nn_rank_size, 0.0f);
                for (int game = 0; game < used_games; ++game) {
                    const GameData& game_data = testing.at(rank)[Random::randInt() % static_cast<int>(testing.at(rank).size())];

                    int temp = 0;
                    int rand_pos = Random::randInt() % 2;
                    int lower_bound = 0;
                    int upper_bound = 1000;

                    if (strength::select_move == "last_50_moves") {
                        lower_bound = static_cast<int>(game_data.scores_.size()) - 50;
                    }
                    if (strength::select_move == "first_50_moves") {
                        upper_bound = 50;
                    }
                    if (strength::select_move == "one_move_per_game") {
                        int select_pos = Random::randInt() % static_cast<int>(game_data.rank_.size());
                        lower_bound = select_pos;
                        upper_bound = select_pos + 1;
                        rand_pos = select_pos % 2;
                    }
                    for (auto& data : game_data.rank_) {
                        float maxElement = data[0]; // Initialize maxElement with the first element
                        int maxIndex = 0;           // Initialize maxIndex with 0

                        if (temp >= lower_bound && temp % 2 == rand_pos) {
                            if (strength::rank_mode == "max_num") {
                                for (int j = 1; j < strength::nn_rank_size; ++j) {
                                    if (data[j] > maxElement) {
                                        maxElement = data[j]; // Update maxElement
                                        maxIndex = j;         // Update maxIndex
                                    }
                                }
                                sum_rank[maxIndex]++;
                            } else if (strength::rank_mode == "max_prob") {
                                for (int j = 0; j < strength::nn_rank_size; ++j) {
                                    sum_rank[j] += data[j];
                                }
                            }
                        }
                        temp++;
                        if (temp >= upper_bound) break;
                    }
                }
                float maxElement = sum_rank[0]; // Initialize maxElement with the first element
                int maxIndex = 0;               // Initialize maxIndex with 0

                // Iterate through the vector starting from the second element
                for (size_t i = 1; i < sum_rank.size(); ++i) {
                    if (sum_rank[i] > maxElement) {
                        maxElement = sum_rank[i]; // Update maxElement
                        maxIndex = i;             // Update maxIndex
                    }
                }

                if (rank == (maxIndex - 1)) { ++correct; }
                if (rank == (maxIndex)) { ++correct_1; }
                if (rank == (maxIndex - 2)) { ++correct1; }
            }
            if (strength::accuracy_mode == "+/-1") {
                correct += correct1;
                correct += correct_1;
            } else if (strength::accuracy_mode == "+1") {
                correct += correct1;
            } else if (strength::accuracy_mode == "-1") {
                correct += correct_1;
            }

            rank_accuracy[rank][used_games - 1] = correct * 1.0f / repeat_times;
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing accuracy ... " << std::endl;
    for (auto& rank_score : testing) { std::cout << "\t" << rank_score.first; } // header
    std::cout << "\tall" << std::endl;
    for (size_t i = 1; i <= max_games; ++i) {
        std::cout << i;
        float avg_accuracy = 0.0f;
        for (auto& rank_score : testing) {
            std::cout << "\t" << rank_accuracy[rank_score.first][i - 1];
            avg_accuracy += rank_accuracy[rank_score.first][i - 1];
        }
        std::cout << "\t" << avg_accuracy / testing.size() << std::endl;
    }
}
void Evaluator::summarizeGamePrediction(const std::vector<std::pair<int, float>>& rank_scores, const std::map<int, std::vector<GameData>>& testing)
{
    const int max_games = 100;
    const int repeat_times = 500;
    std::map<int, std::vector<float>> rank_accuracy;
    for (auto& rank_score : rank_scores) {
        int rank = rank_score.first;
        rank_accuracy[rank].resize(max_games, 0.0f);
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Predicting rank " << rank << " ... " << std::endl;

        for (int used_games = 1; used_games <= max_games; ++used_games) {
            int correct = 0;
            int correct1 = 0;
            int correct_1 = 0;
            for (int i = 0; i < repeat_times; ++i) {
                float score = 0;
                int num_position = 0;
                for (int game = 0; game < used_games; ++game) {
                    const GameData& game_data = testing.at(rank)[Random::randInt() % static_cast<int>(testing.at(rank).size())];
                    int rand_pos = Random::randInt() % 2;
                    int temp = 0;
                    int lower_bound = 0;
                    int upper_bound = 1000;

                    if (strength::select_move == "last_50_moves") {
                        lower_bound = static_cast<int>(game_data.scores_.size()) - 50;
                    }
                    if (strength::select_move == "first_50_moves") {
                        upper_bound = 50;
                    }
                    if (strength::select_move == "one_move_per_game") {
                        int select_pos = Random::randInt() % static_cast<int>(game_data.scores_.size());
                        lower_bound = select_pos;
                        upper_bound = select_pos + 1;
                        rand_pos = select_pos % 2;
                    }

                    for (auto& s : game_data.scores_) {
                        if (temp >= lower_bound && temp % 2 == rand_pos) {
                            score += s;
                            num_position++;
                        }
                        temp++;
                        if (temp >= upper_bound) break;
                    }
                }

                float average_score = score / num_position;
                int prediected_rank = std::min_element(rank_scores.begin(), rank_scores.end(), [average_score](const std::pair<int, float>& a, const std::pair<int, float>& b) { return std::abs(average_score - a.second) < std::abs(average_score - b.second); })->first;
                if (prediected_rank == rank) { correct++; }
                if (prediected_rank == rank + 1) { correct1++; }
                if (prediected_rank == rank - 1) { correct_1++; }
            }
            if (strength::accuracy_mode == "+/-1") {
                correct += correct1;
                correct += correct_1;
            } else if (strength::accuracy_mode == "+1") {
                correct += correct1;
            } else if (strength::accuracy_mode == "-1") {
                correct += correct_1;
            }
            rank_accuracy[rank][used_games - 1] = correct * 1.0f / repeat_times;
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing accuracy ... " << std::endl;
    for (auto& rank_score : rank_scores) { std::cout << "\t" << rank_score.first; } // header
    std::cout << "\tall" << std::endl;
    for (size_t i = 1; i <= max_games; ++i) {
        std::cout << i;
        float avg_accuracy = 0.0f;
        for (auto& rank_score : rank_scores) {
            std::cout << "\t" << rank_accuracy[rank_score.first][i - 1];
            avg_accuracy += rank_accuracy[rank_score.first][i - 1];
        }
        std::cout << "\t" << avg_accuracy / rank_scores.size() << std::endl;
    }
}

} // namespace strength
