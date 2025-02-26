#pragma once

#include "environment.h"
#include "mode_handler.h"
#include "st_configuration.h"
#include "strength_network.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace strength {

class StModeHandler : public minizero::console::ModeHandler {
public:
    StModeHandler();
    void loadNetwork(const std::string& nn_file_name, int gpu_id = 0);

private:
    void setDefaultConfiguration(minizero::config::ConfigureLoader& cl) override { strength::setConfiguration(cl); }
    void runConsole() override;
    void runSelfPlay() override;
    void runMCTSAccuracy();
    std::map<int, std::vector<std::pair<float, float>>> calculatePosStrength(const std::vector<EnvironmentLoader>& env_loaders);
    std::map<int, std::vector<std::pair<float, float>>> calculatePosStrength(const EnvironmentLoader& env_loader);
    void runZeroTrainingName() override;
    void runEvaluator();
    std::shared_ptr<StrengthNetwork> network_;
    std::string getNetworkAbbeviation() const;
};

} // namespace strength
