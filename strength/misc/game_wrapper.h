#pragma once

#include "environment.h"
#include <string>
#include <vector>

namespace strength {

std::vector<EnvironmentLoader> loadGames(const std::string& file_name);
EnvironmentLoader loadGame(const std::string& file_content);
std::vector<float> calculateFeatures(const Environment& env, minizero::utils::Rotation rotation = minizero::utils::Rotation::kRotationNone);
std::vector<float> calculateFeatures(const EnvironmentLoader& env_loader, const int& pos, minizero::utils::Rotation rotation = minizero::utils::Rotation::kRotationNone);
int getRank(const EnvironmentLoader& env_loader);

} // namespace strength
