#pragma once

#include "alphazero_network.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace strength {

class StrengthNetworkOutput : public minizero::network::AlphaZeroNetworkOutput {
public:
    std::vector<float> rank_;
    std::vector<float> rank_logits_;
    float score_;
    float weight_;

    StrengthNetworkOutput(int policy_size, int rank_size)
        : AlphaZeroNetworkOutput(policy_size)
    {
        rank_.resize(rank_size, 0.0f);
        rank_logits_.resize(rank_size, 0.0f);
        score_ = 0.0f;
        weight_ = 0.0f;
    }
};

class StrengthNetwork : public minizero::network::AlphaZeroNetwork {
public:
    StrengthNetwork() : AlphaZeroNetwork() {}

    void loadModel(const std::string& nn_file_name, const int gpu_id) override
    {
        AlphaZeroNetwork::loadModel(nn_file_name, gpu_id);
        std::vector<torch::jit::IValue> dummy;
        rank_size_ = (network_type_name_ == "rank" ? network_.get_method("get_rank_size")(dummy).toInt() : 0);
    }

    std::vector<std::shared_ptr<minizero::network::NetworkOutput>> forward()
    {
        assert(batch_size_ > 0);
        auto forward_result = network_.forward(std::vector<torch::jit::IValue>{torch::cat(tensor_input_).to(getDevice())}).toGenericDict();

        auto policy_output = forward_result.at("policy").toTensor().to(at::kCPU);
        auto policy_logits_output = forward_result.at("policy_logit").toTensor().to(at::kCPU);
        auto value_output = forward_result.at("value").toTensor().to(at::kCPU);
        auto rank_output = (forward_result.contains("rank") ? forward_result.at("rank").toTensor().to(at::kCPU) : torch::zeros(0));
        auto rank_logits_output = (forward_result.contains("rank_logit") ? forward_result.at("rank_logit").toTensor().to(at::kCPU) : torch::zeros(0));
        auto score_output = (forward_result.contains("score") ? forward_result.at("score").toTensor().to(at::kCPU) : torch::zeros(0));
        auto weight_output = (forward_result.contains("weight") ? forward_result.at("weight").toTensor().to(at::kCPU) : torch::zeros(0));
        assert(policy_output.numel() == batch_size_ * getActionSize());
        assert(policy_logits_output.numel() == batch_size_ * getActionSize());
        assert(value_output.numel() == batch_size_ * getDiscreteValueSize());
        assert(!forward_result.contains("rank") || (forward_result.contains("rank") && rank_output.numel() == batch_size_ * getRankSize()));
        assert(!forward_result.contains("rank_logit") || (forward_result.contains("rank_logit") && rank_logits_output.numel() == batch_size_ * getRankSize()));
        assert(!forward_result.contains("score") || (forward_result.contains("score") && score_output.numel() == batch_size_));
        assert(!forward_result.contains("weight") || (forward_result.contains("weight") && weight_output.numel() == batch_size_));

        const int policy_size = getActionSize();
        const int rank_size = getRankSize();
        std::vector<std::shared_ptr<minizero::network::NetworkOutput>> network_outputs;
        for (int i = 0; i < batch_size_; ++i) {
            network_outputs.emplace_back(std::make_shared<StrengthNetworkOutput>(policy_size, rank_size));
            auto strength_network_output = std::static_pointer_cast<StrengthNetworkOutput>(network_outputs.back());

            // policy & policy logits
            std::copy(policy_output.data_ptr<float>() + i * policy_size,
                      policy_output.data_ptr<float>() + (i + 1) * policy_size,
                      strength_network_output->policy_.begin());
            std::copy(policy_logits_output.data_ptr<float>() + i * policy_size,
                      policy_logits_output.data_ptr<float>() + (i + 1) * policy_size,
                      strength_network_output->policy_logits_.begin());

            // value
            if (getDiscreteValueSize() == 1) {
                strength_network_output->value_ = value_output[i].item<float>();
            } else {
                int start_value = -getDiscreteValueSize() / 2;
                strength_network_output->value_ = std::accumulate(value_output.data_ptr<float>() + i * getDiscreteValueSize(),
                                                                  value_output.data_ptr<float>() + (i + 1) * getDiscreteValueSize(),
                                                                  0.0f,
                                                                  [&start_value](const float& sum, const float& value) { return sum + value * start_value++; });
                strength_network_output->value_ = minizero::utils::invertValue(strength_network_output->value_);
            }

            // rank & rank logits
            if (forward_result.contains("rank")) {
                std::copy(rank_output.data_ptr<float>() + i * rank_size,
                          rank_output.data_ptr<float>() + (i + 1) * rank_size,
                          strength_network_output->rank_.begin());
                std::copy(rank_logits_output.data_ptr<float>() + i * rank_size,
                          rank_logits_output.data_ptr<float>() + (i + 1) * rank_size,
                          strength_network_output->rank_logits_.begin());
            }

            // score
            if (forward_result.contains("score")) { strength_network_output->score_ = score_output[i].item<float>(); }

            // weight
            if (forward_result.contains("weight")) { strength_network_output->weight_ = weight_output[i].item<float>(); }
        }

        clear();
        return network_outputs;
    }

    inline int getRankSize() const { return rank_size_; }

private:
    int rank_size_;
};

} // namespace strength
