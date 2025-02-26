#pragma once

#include "configure_loader.h"
#include <string>
#include <vector>
namespace strength {

extern bool actor_select_action_by_ssa;
extern bool actor_select_action_by_bt;
extern int nn_rank_size;
extern int bt_num_batch_size;
extern int bt_num_rank_per_batch;
extern int bt_num_position_per_rank;
extern bool bt_use_weight;
extern bool bt_use_same_game_per_rank;
extern bool bt_add_non_people;
extern std::string training_sgf_dir;
extern std::string testing_sgf_dir;
extern std::string candidate_sgf_dir;
extern std::string evaluator_mode;
extern std::string rank_mode;
extern std::string accuracy_mode;
extern std::string select_move;
extern float s_weight;
extern std::vector<float> cand_strength;
extern std::vector<float> temp_for_mcts_ssa_accuracy;

//for chess
extern int nn_rank_min_elo;
extern int nn_rank_elo_interval;

void setConfiguration(minizero::config::ConfigureLoader& cl);

} // namespace strength
