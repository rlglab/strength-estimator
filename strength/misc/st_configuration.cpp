#include "st_configuration.h"
#include "configuration.h"
#include <vector>
namespace strength {

using namespace minizero;

bool actor_select_action_by_ssa = false;
bool actor_select_action_by_bt = false;
int nn_rank_size = 9;
int bt_num_batch_size = 32;
int bt_num_rank_per_batch = 9;
int bt_num_position_per_rank = 7;
bool bt_use_weight = false;
bool bt_use_same_game_per_rank = true;
bool bt_add_non_people = false;
std::string training_sgf_dir = "training_sgf_go";
std::string testing_sgf_dir = "query_sgf_go";
std::string candidate_sgf_dir = "candidate_sgf_go";
std::string evaluator_mode = "game_prediction";
std::string rank_mode = "max_prob";
std::string accuracy_mode = "+/-0";
std::string select_move = "all_moves";
float s_weight = 2.0;
std::vector<float> cand_strength(500, 0.0f);
std::vector<float> temp_for_mcts_ssa_accuracy = {-2.0, -1, -0.6, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.5, 0.6, 1.0, 2.0};

//for chess
int nn_rank_min_elo = 1000;
int nn_rank_elo_interval = 200;

void setConfiguration(config::ConfigureLoader& cl)
{
    config::setConfiguration(cl);

    cl.addParameter("actor_select_action_by_ssa", actor_select_action_by_ssa, "", "Actor");
    cl.addParameter("actor_select_action_by_bt", actor_select_action_by_bt, "", "Actor");
    cl.addParameter("nn_rank_size", nn_rank_size, "", "Strength");
    cl.addParameter("bt_num_batch_size", bt_num_batch_size, "batch size in bt training = bt_num_batch_size * bt_num_rank_per_batch * bt_num_position_per_rank", "Strength");
    cl.addParameter("bt_num_rank_per_batch", bt_num_rank_per_batch, "batch size in bt training = bt_num_batch_size * bt_num_rank_per_batch * bt_num_position_per_rank", "Strength");
    cl.addParameter("bt_num_position_per_rank", bt_num_position_per_rank, "batch size in bt training = bt_num_batch_size * bt_num_rank_per_batch * bt_num_position_per_rank", "Strength");
    cl.addParameter("bt_use_weight", bt_use_weight, "", "Strength");
    cl.addParameter("bt_use_same_game_per_rank", bt_use_same_game_per_rank, "", "Strength");
    cl.addParameter("bt_add_non_people", bt_add_non_people, "when training bt, add the non-people move to be the lowest rank", "Strength");
    cl.addParameter("training_sgf_dir", training_sgf_dir, "", "Strength");
    cl.addParameter("testing_sgf_dir", testing_sgf_dir, "", "Strength");
    cl.addParameter("candidate_sgf_dir", candidate_sgf_dir, "", "Strength");
    cl.addParameter("evaluator_mode", evaluator_mode, "game_prediction/move_prediction", "Strength");
    cl.addParameter("rank_mode", rank_mode, "max_prob/max_num", "Strength");
    cl.addParameter("accuracy_mode", accuracy_mode, "+/-0,+/-1,+1,-1", "Strength");
    cl.addParameter("select_move", select_move, "all_moves/first_50_moves/last_50_moves/one_move_per_game", "Strength");
    cl.addParameter("s_weight", s_weight, "weight for puct value_s", "Strength");
	
#if CHESS
	cl.addParameter("nn_rank_min_elo", nn_rank_min_elo, "min elo", "Strength");
    cl.addParameter("nn_rank_elo_interval", nn_rank_elo_interval, "size of elo interval", "Strength");
#endif
}

} // namespace strength
