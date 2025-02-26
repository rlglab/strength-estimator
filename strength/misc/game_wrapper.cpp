#include "game_wrapper.h"
#include "sgf_loader.h"
#include "st_configuration.h"
#include <algorithm>
#include <string>
#include <vector>

namespace strength {

using namespace minizero;
using namespace minizero::utils;
std::vector<EnvironmentLoader> loadGames(const std::string& file_name)
{
    std::vector<EnvironmentLoader> env_loaders;
    std::ifstream fin(file_name, std::ifstream::in);
    for (std::string content; std::getline(fin, content);) {
        EnvironmentLoader env_loader = loadGame(content);
        if (env_loader.getActionPairs().empty()) { continue; }
        env_loaders.push_back(env_loader);
    }
    return env_loaders;
}
EnvironmentLoader loadGame(const std::string& file_content)
{
#if GO
    EnvironmentLoader env_loader;
    // if (env_loader.loadFromString(file_content)) { return env_loader; }

    if (file_content.empty()) { return EnvironmentLoader(); }
    if (file_content.find("(") == std::string::npos) { return EnvironmentLoader(); }

    SGFLoader sgf_loader;
    if (!sgf_loader.loadFromString(file_content)) { return EnvironmentLoader(); }
    if (std::stoi(sgf_loader.getTags().at("SZ")) != 19) { return EnvironmentLoader(); }

    env_loader.reset();
    env_loader.addTag("SZ", sgf_loader.getTags().at("SZ"));
    env_loader.addTag("KM", sgf_loader.getTags().at("KM"));
    env_loader.addTag("RE", std::to_string(sgf_loader.getTags().at("RE")[0] == 'B' ? 1.0f : -1.0f));
    env_loader.addTag("PB", sgf_loader.getTags().at("PB"));
    env_loader.addTag("PW", sgf_loader.getTags().at("PW"));
    env_loader.addTag("BR", sgf_loader.getTags().at("BR"));
    env_loader.addTag("WR", sgf_loader.getTags().at("WR"));
    for (auto& action_string : sgf_loader.getActions()) { env_loader.addActionPair(Action(action_string.first, std::stoi(sgf_loader.getTags().at("SZ"))), action_string.second); }
    return env_loader;
#else
    EnvironmentLoader env_loader;
    env_loader.loadFromString(file_content);
    return env_loader;
#endif
}

std::vector<float> calculateFeatures(const Environment& env, minizero::utils::Rotation rotation /* = minizero::utils::Rotation::kRotationNone */)
{
    return env.getFeatures(rotation);
}

std::vector<float> calculateFeatures(const EnvironmentLoader& env_loader, const int& pos, minizero::utils::Rotation rotation /* = minizero::utils::Rotation::kRotationNone */)
{
    return env_loader.getFeatures(pos, rotation);
}

int getRank(const EnvironmentLoader& env_loader)
{
#if GO
    std::string rank_str = env_loader.getTag("BR");
    if (rank_str.empty()) { return 0; }
    int rank = std::stoi(rank_str);
    if (rank_str.find_first_of("kK") != std::string::npos) {
        rank = -rank + 1;

        // 3-5k,1-2k,1-9D
        if (rank <= -2)
            rank = -1;
        else if (rank <= 0)
            rank = 0;
    }
    return rank;
#elif CHESS
    // 0 under min elo
    // 1 ~ rank size - 2: min elo + interval * rank
    // rank size - 1: above max elo
    std::string rank_str = env_loader.getTag("BR");
    int rank = std::stoi(rank_str);
    int ret = (rank - strength::nn_rank_min_elo) / strength::nn_rank_elo_interval;
    return ret;
#else
    // TODO: each game should implement its own getRank
    return 0;
#endif
}

} // namespace strength
