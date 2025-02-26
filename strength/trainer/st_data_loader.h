#pragma once

#include "data_loader.h"
#include "st_configuration.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strength {

class GamePosition {
public:
    int rank_;
    int env_id_;
    int pos_;

    GamePosition() {}
    GamePosition(int rank, int env_id, int pos) : rank_(rank), env_id_(env_id), pos_(pos) {}

    inline bool operator==(const GamePosition& gp) const { return (rank_ == gp.rank_) && (env_id_ == gp.env_id_) && (pos_ == gp.pos_); }
    inline bool operator!=(const GamePosition& gp) const { return !(*this == gp); }
};

class StBatchDataPtr : public minizero::learner::BatchDataPtr {
public:
    float* rank_;
};

class StDataLoaderSharedData : public minizero::learner::DataLoaderSharedData {
public:
    void createDataPtr() override { data_ptr_ = std::make_shared<StBatchDataPtr>(); }
    inline std::shared_ptr<StBatchDataPtr> getDataPtr() { return std::static_pointer_cast<StBatchDataPtr>(data_ptr_); }

    std::vector<GamePosition> bt_game_positions_;
    std::map<int, int> rank_label_map_;
    std::map<int, std::vector<EnvironmentLoader>> env_loaders_map_;
};

class StDataLoaderThread : public minizero::learner::DataLoaderThread {
public:
    StDataLoaderThread(int id, std::shared_ptr<minizero::utils::BaseSharedData> shared_data)
        : DataLoaderThread(id, shared_data) {}

protected:
    bool addEnvironmentLoader() override;
    bool sampleData() override;

    inline std::shared_ptr<StDataLoaderSharedData> getSharedData() { return std::static_pointer_cast<StDataLoaderSharedData>(shared_data_); }

private:
    void setAlphaZeroTrainingData(int batch_index);
    void setRankTrainingData(int batch_index);
    void setBTTrainingData(int batch_index);
    GamePosition sampleTrainingData();
};

class StDataLoader : public minizero::learner::DataLoader {
public:
    StDataLoader(const std::string& conf_file_name);
    void initialize() override;
    void loadDataFromFile(const std::string& file_name) override;
    void sampleData() override;

    void createSharedData() override { shared_data_ = std::make_shared<StDataLoaderSharedData>(); }
    std::shared_ptr<minizero::utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<StDataLoaderThread>(id, shared_data_); }
    inline std::shared_ptr<StDataLoaderSharedData> getSharedData() { return std::static_pointer_cast<StDataLoaderSharedData>(shared_data_); }

private:
    void allocateBTGamePositions();
};

} // namespace strength
