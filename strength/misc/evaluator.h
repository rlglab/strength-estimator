#pragma once

#include "environment.h"
#include "network.h"
#include "paralleler.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strength {

class GameData {
public:
    std::vector<float> scores_;
    std::vector<float> weights_;
    std::vector<float> value_;
    std::vector<std::vector<float>> rank_;
    std::vector<std::vector<float>> policy_;
    EnvironmentLoader env_loader_;
};

class EvaluatorSharedData : public minizero::utils::BaseSharedData {
public:
    int sgf_index_;
    std::mutex mutex_;
    std::vector<std::string> sgfs_;
    std::map<int, std::vector<GameData>> env_loaders_map_;
    std::vector<std::unique_ptr<std::mutex>> network_mutexes_;
    std::vector<std::shared_ptr<minizero::network::Network>> networks_;

    int getNextSgfIndex();
};

class EvaluatorSlaveThread : public minizero::utils::BaseSlaveThread {
public:
    EvaluatorSlaveThread(int id, std::shared_ptr<minizero::utils::BaseSharedData> shared_data)
        : BaseSlaveThread(id, shared_data) {}

    void initialize() override;
    void runJob() override;
    bool isDone() override { return false; }

private:
    inline std::shared_ptr<EvaluatorSharedData> getSharedData() { return std::static_pointer_cast<EvaluatorSharedData>(shared_data_); }
};

class Evaluator : public minizero::utils::BaseParalleler {
public:
    Evaluator() {}

    void run();
    void initialize() override;
    void summarize() override {}

private:
    void createNeuralNetworks();
    void runGamePrediction();
    void runMovePrediction();
    void evaluateGames(const std::string& directory);
    void summarizeGamePrediction(const std::map<int, std::vector<GameData>>& testing);
    void summarizeGamePrediction(const std::vector<std::pair<int, float>>& rank_scores, const std::map<int, std::vector<GameData>>& testing);

    void createSharedData() override { shared_data_ = std::make_shared<EvaluatorSharedData>(); }
    std::shared_ptr<minizero::utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<EvaluatorSlaveThread>(id, shared_data_); }
    inline std::shared_ptr<EvaluatorSharedData> getSharedData() { return std::static_pointer_cast<EvaluatorSharedData>(shared_data_); }

    std::map<int, std::vector<GameData>> testing_env_loaders_map_;
    std::map<int, std::vector<GameData>> candidate_env_loaders_map_;
};

} // namespace strength
