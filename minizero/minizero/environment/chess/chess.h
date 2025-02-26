#pragma once

#include "base_env.h"
#include <map>
#include <string>
#include <vector>
#include "position.h"
#include <bitset>
#include <iostream>

namespace minizero::env::chess {

const std::string kChessName = "chess";
const int kChessNumPlayer = 2;
const int kChessBoardSize = 8;

std::string getChessActionName(int action_id);

// typedef Move ChessAction;
class ChessAction : public BaseAction {
public:
    ChessAction() : BaseAction() {}
    ChessAction(int action_id, Player player) : BaseAction(action_id, player) {}
    ChessAction(const std::vector<std::string>& action_string_args)
    {
        assert(action_string_args.size() == 2);
        assert(action_string_args[0].size() == 1);
        player_ = charToPlayer(action_string_args[0][0]);
        assert(static_cast<int>(player_) > 0 && static_cast<int>(player_) <= kNumPlayer); // assume kPlayer1 == 1, kPlayer2 == 2, ...
        auto move_ = Move(action_string_args[1], player_ == Player::kPlayer2);
        action_id_ = move_.as_nn_index(0);
    }

    inline Player nextPlayer() const override { return getNextPlayer(player_, kChessNumPlayer); }
    inline std::string toConsoleString() const override
    {
        return Move(action_id_, player_ == Player::kPlayer2).as_string();
        
    }
    Move move() const { return Move(action_id_); }

   
};

class ChessEnv : public BaseBoardEnv<ChessAction> {
public:
    ChessEnv() : BaseBoardEnv<ChessAction>(kChessBoardSize) { reset(); }

    void reset() override;
    bool act(const ChessAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<ChessAction> getLegalActions() const override;
    void setLegalAction();
    bool isLegalAction(const ChessAction& action) const override;
    bool isTerminal() const override;
    float getReward() const override { return 0.0f; }
    float getEvalScore(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const ChessAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::string toString() const override;
    inline std::string name() const override { return kChessName; }
    inline int getNumPlayer() const override { return kChessNumPlayer; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    inline int getNumInputChannels() const override { return 119; }
    inline int getPolicySize() const override { return 1858; }
    inline int getNumActionFeatureChannels() const override { return 6; }

private:
    PositionHistory history_;
    GameResult winner_ = GameResult::UNDECIDED;
    std::bitset<1858> legal_action_;
};

class ChessEnvLoader : public BaseBoardEnvLoader<ChessAction, ChessEnv> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::vector<float> getValue(const int pos) const { return {getReturn()}; }
    inline std::string name() const override { return kChessName; }
    bool loadFromString(const std::string& content) override;
    inline int getPolicySize() const override { return 1858; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
};

} // namespace minizero::env::chess
