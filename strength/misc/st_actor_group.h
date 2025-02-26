#pragma once

#include "actor_group.h"
#include <memory>
#include <vector>

namespace strength {

class STSharedData : public minizero::actor::ThreadSharedData {
public:
};

class STSlaveThread : public minizero::actor::SlaveThread {
public:
    STSlaveThread(int id, std::shared_ptr<minizero::utils::BaseSharedData> shared_data)
        : SlaveThread(id, shared_data) {}

private:
    bool doCPUJob() override;
    void doGPUJob() override;
};

class STActorGroup : public minizero::actor::ActorGroup {
public:
    STActorGroup() {}

    void initialize() override;
    void step();

    inline std::vector<std::shared_ptr<minizero::actor::BaseActor>>& getActors() { return getSharedData()->actors_; }

private:
    void createNeuralNetworks() override;
    void createActors() override;

    void createSharedData() override { shared_data_ = std::make_shared<STSharedData>(); }
    std::shared_ptr<minizero::utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<STSlaveThread>(id, shared_data_); }
    inline std::shared_ptr<STSharedData> getSharedData() { return std::static_pointer_cast<STSharedData>(shared_data_); }
};

} // namespace strength
