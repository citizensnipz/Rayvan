#pragma once

#include "rayvan_emc/tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rayvan::emc {

struct TokenStreamMetadata {
    std::string split_id;
    std::uint64_t token_count = 0;
    std::uint64_t fingerprint = 0;
};

class TokenStream final {
public:
    static TokenStream load(const std::filesystem::path& path);
    static void save(
        const std::filesystem::path& path,
        const std::vector<std::int32_t>& tokens,
        const std::string& split_id);

    [[nodiscard]] const TokenStreamMetadata& metadata() const noexcept { return metadata_; }
    [[nodiscard]] const std::vector<std::int32_t>& tokens() const noexcept { return tokens_; }
    [[nodiscard]] Tensor sample_batch(
        std::int64_t batch_size,
        std::int64_t sequence_length,
        torch::Generator& generator,
        const torch::Device& device) const;

private:
    TokenStreamMetadata metadata_;
    std::vector<std::int32_t> tokens_;
};

std::uint64_t token_fingerprint(const std::vector<std::int32_t>& tokens, const std::string& split_id);

}  // namespace rayvan::emc
