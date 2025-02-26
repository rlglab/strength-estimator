#include "chess.h"
#include "random.h"
#include <utility>

namespace minizero::env::chess {
using namespace minizero::utils;

std::string getChessActionName(int action_id)
{
    return Move(action_id).as_string();
}

void ChessEnv::reset()
{
    InitializeMagicBitboards();
    turn_ = Player::kPlayer1;
    actions_.clear();
    winner_ = GameResult::UNDECIDED;
    history_.Reset(ChessBoard::kStartposBoard, 0, 0);
    setLegalAction();
}

bool ChessEnv::act(const ChessAction& action)
{
    if ((action.getPlayer() == Player::kPlayer2) != history_.Last().IsBlackToMove()) {
        return false;
    }
    if (!isLegalAction(action)) {
        return false;
    }
    actions_.push_back(action);
    turn_ = (turn_ == Player::kPlayer1 ? Player::kPlayer2 : Player::kPlayer1);
    history_.Append(action.move());
    setLegalAction();
    winner_ = history_.ComputeGameResult();
   
    return true;
}
bool ChessEnv::act(const std::vector<std::string>& action_string_args)
{
    return act(ChessAction(action_string_args));
}
std::vector<ChessAction> ChessEnv::getLegalActions() const
{
    std::vector<ChessAction> ret_actions;
    for (size_t i = 0; i < 1858; ++i) {
        if (legal_action_.test(i)) {
            ret_actions.emplace_back(ChessAction(i, turn_));
        }
    }
    return ret_actions;
}
void ChessEnv::setLegalAction()
{
    legal_action_.reset();
    auto board = history_.Last().GetBoard();
    MoveList ml = board.GenerateLegalMoves();
    for (auto& m : ml) {
        legal_action_.set(m.as_nn_index(0));
    }
}
bool ChessEnv::isLegalAction(const ChessAction& action) const
{
    return legal_action_.test(action.getActionID());
}


bool ChessEnv::isTerminal() const
{
    return winner_ != GameResult::UNDECIDED;
   
}
float ChessEnv::getEvalScore(bool is_resign) const
{
    if (is_resign) {
        if (getNextPlayer(turn_, kChessNumPlayer) == Player::kPlayer1) {
            return 1.0f;
        } else {
            return -1.0f;
        }
    } else {
        switch (winner_) {
            case GameResult::WHITE_WON: return 1.0f;
            case GameResult::BLACK_WON: return -1.0f;
            default: return 0.0f;
        }
        
    }
}
struct InputPlane {
    InputPlane() = default;
    void SetAll() { mask = ~0ull; }
    void Fill(float val)
    {
        SetAll();
        value = val;
    }
    std::uint64_t mask = 0ull;
    float value = 1.0f;
};
using InputPlanes = std::vector<InputPlane>;
const int kMoveHistory = 8;
const int kPlanesPerBoard = 14;
const int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
InputPlanes EncodePositionForNN(const PositionHistory& history)
{
    InputPlanes result(kAuxPlaneBase + 7);

    {
        const ChessBoard& board = history.Last().GetBoard();
        const bool we_are_black = board.flipped();
        if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
        if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
        if (board.castlings().they_can_000()) result[kAuxPlaneBase + 2].SetAll();
        if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
        if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
        result[kAuxPlaneBase + 5].Fill(static_cast<float>(history.Last().GetRule50Ply()) / 100.0f);
        result[kAuxPlaneBase + 6].Fill(static_cast<float>(history.Last().GetGamePly()) / 200.0f);  
    }

    bool flip = false;
    int history_idx = history.GetLength() - 1;
    for (int i = 0; i < kMoveHistory; ++i, flip = !flip, --history_idx) {
        if (history_idx < 0) break;
        const Position& position = history.GetPositionAt(history_idx);
        const ChessBoard& board =
            flip ? position.GetThemBoard() : position.GetBoard();

        const int base = i * kPlanesPerBoard;
        result[base + 0].mask = (board.ours() & board.pawns()).as_int();
        result[base + 1].mask = (board.ours() & board.knights()).as_int();
        result[base + 2].mask = (board.ours() & board.bishops()).as_int();
        result[base + 3].mask = (board.ours() & board.rooks()).as_int();
        result[base + 4].mask = (board.ours() & board.queens()).as_int();
        result[base + 5].mask = (board.ours() & board.kings()).as_int();

        result[base + 6].mask = (board.theirs() & board.pawns()).as_int();
        result[base + 7].mask = (board.theirs() & board.knights()).as_int();
        result[base + 8].mask = (board.theirs() & board.bishops()).as_int();
        result[base + 9].mask = (board.theirs() & board.rooks()).as_int();
        result[base + 10].mask = (board.theirs() & board.queens()).as_int();
        result[base + 11].mask = (board.theirs() & board.kings()).as_int();

        const int repetitions = position.GetRepetitions();
        if (repetitions >= 1) result[base + 12].SetAll();
        if (repetitions >= 2) result[base + 13].SetAll();
    }

    return result;
}
std::vector<float> OutputFeature(InputPlanes ip)
{
    std::vector<float> ret;
    for (auto& it : ip) {
        uint64_t idx = 1ull;
        while (idx) {
            if (it.mask & idx) {
                ret.push_back(it.value);
            } else {
                ret.push_back(0.0f);
            }
            idx <<= 1;
        }
    }
    return ret;
}
std::vector<float> ChessEnv::getFeatures(utils::Rotation rotation) const
{
    /* 119 channels:
        0~112. own/opponent history
            x~x+5. P1's piece
            x+6~x+11. P2's piece
            x+12~x+13. repetitions
        113~116. own/opponent castling
        117. color(0: white, 1: black)
        118. Rule 50
        119. step count
    */
    auto planes = EncodePositionForNN(history_);
    return OutputFeature(planes);
}
std::vector<float> ChessEnv::getActionFeatures(const ChessAction& action, utils::Rotation rotation) const
{
    /* 6 channels:
        0. start
        1. end
        2~5. Promotion
            2. No Promotion/Knight
            3. Queen
            4. Rook
            5. Bishop
    */
    auto plane_size = 64;
    std::vector<float> ret(6 * plane_size, 0.0f);
    auto m = action.move();
    ret[m.from().as_int()] = 1.0f;
    ret[plane_size + m.to().as_int()] = 1.0f;
    std::fill(ret.begin() + plane_size * ((std::uint8_t)m.promotion() + 2), ret.begin() + plane_size * ((std::uint8_t)m.promotion() + 3), 1.0f);
    return ret;
}
std::string ChessEnv::toString() const
{
    return history_.Last().GetWhiteBoard().DebugString();
}

std::vector<float> ChessEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation) const
{
    auto plane_size = 64;
    std::vector<float> action_features(6 * plane_size, 0.0f);
    ChessAction action;
    if (pos < static_cast<int>(action_pairs_.size())) {
        action = action_pairs_[pos].first;
    } else {
        int action_id = utils::Random::randInt() % getPolicySize();
        action = ChessAction(action_id, Player::kPlayer1);
    }
    auto m = action.move();
    action_features[m.from().as_int()] = 1.0f;
    action_features[plane_size + m.to().as_int()] = 1.0f;
    std::fill(action_features.begin() + plane_size * ((std::uint8_t)m.promotion() + 2), action_features.begin() + plane_size * ((std::uint8_t)m.promotion() + 3), 1.0f);

    return action_features;  
}

bool ChessEnvLoader::loadFromString(const std::string& content)
{
    reset();
    sgf_content_ = content;
    std::string key, value;
    int state = '(';
    bool accept_move = false;
    bool escape_next = false;
    bool is_black = false;
    for (char c : content) {
        switch (state) {
            case '(': // wait until record start
                if (!accept_move) {
                    accept_move = (c == '(');
                } else {
                    state = (c == ';') ? c : 'x';
                    accept_move = false;
                }
                break;
            case ';': // store key
                if (c == ';') {
                    accept_move = true;
                } else if (c == '[' || c == ')') {
                    state = c;
                } else if (std::isgraph(c)) {
                    key += c;
                }
                break;
            case '[': // store value
                if (c == '\\' && !escape_next) {
                    escape_next = true;
                } else if (c != ']' || escape_next) {
                    value += c;
                    escape_next = false;
                } else { // ready to store key-value pair
                    if (accept_move) {
                        int action_id = value.size() && std::isdigit(value[0]) ? std::stoi(value) : Move(value, is_black).as_nn_index(0);
                        is_black = !is_black;
                        action_pairs_.emplace_back().first = ChessAction(action_id, charToPlayer(key[0]));
                        accept_move = false;
                    } else if (action_pairs_.size()) {
                        action_pairs_.back().second[key] = std::move(value);
                    } else {
                        tags_[key] = std::move(value);
                    }
                    key.clear();
                    value.clear();
                    state = ';';
                }
                break;
            case ')': // end of record, do nothing
                break;
        }
    }
    return state == ')';
}

} // namespace minizero::env::chess
