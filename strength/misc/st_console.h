#pragma once

#include "console.h"
#include "rotation.h"
#include <map>
#include <string>
#include <utility>
#include <vector>
namespace strength {

class StConsole : public minizero::console::Console {
public:
    StConsole();

    void initialize() override;

protected:
    void cmdGoguiAnalyzeCommands(const std::vector<std::string>& args);
    void cmdStrength(const std::vector<std::string>& args);

    void calculatePolicyValue(std::vector<float>& policy, float& value, minizero::utils::Rotation rotation = minizero::utils::Rotation::kRotationNone) override;
    std::map<int, std::vector<std::pair<float, float>>> calculatePosStrength(const std::vector<EnvironmentLoader>& env_loaders);
    std::map<int, std::vector<std::pair<float, float>>> calculatePosStrength(const EnvironmentLoader& env_loader);
};

} // namespace strength
