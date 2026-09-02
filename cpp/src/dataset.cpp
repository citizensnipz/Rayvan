#include "rayvan_emc/training/dataset.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace rayvan::emc {
namespace {

constexpr std::array<char, 8> magic{'R', 'V', 'T', 'O', 'K', 'E', 'N', '1'};
constexpr std::uint32_t version = 1;

template <typename T>
void write_value(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
T read_value(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("truncated token stream");
    return value;
}

void fnv_byte(std::uint64_t& hash, std::uint8_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
}

}  // namespace

std::uint64_t token_fingerprint(
    const std::vector<std::int32_t>& tokens,
    const std::string& split_id) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : split_id) fnv_byte(hash, static_cast<std::uint8_t>(value));
    fnv_byte(hash, 0);
    for (const auto token : tokens) {
        const auto value = static_cast<std::uint32_t>(token);
        for (unsigned shift = 0; shift < 32; shift += 8) fnv_byte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
    return hash;
}

void TokenStream::save(
    const std::filesystem::path& path,
    const std::vector<std::int32_t>& tokens,
    const std::string& split_id) {
    if (tokens.size() < 2) throw std::invalid_argument("token stream must contain at least two tokens");
    if (split_id.empty() || split_id.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid token stream split ID");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write token stream: " + path.string());
    stream.write(magic.data(), magic.size());
    write_value(stream, version);
    write_value(stream, static_cast<std::uint64_t>(tokens.size()));
    write_value(stream, token_fingerprint(tokens, split_id));
    write_value(stream, static_cast<std::uint32_t>(split_id.size()));
    stream.write(split_id.data(), static_cast<std::streamsize>(split_id.size()));
    stream.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
    if (!stream) throw std::runtime_error("failed writing token stream: " + path.string());
}

TokenStream TokenStream::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read token stream: " + path.string());
    std::array<char, 8> observed{};
    stream.read(observed.data(), observed.size());
    if (observed != magic) throw std::runtime_error("invalid token stream magic");
    if (read_value<std::uint32_t>(stream) != version) throw std::runtime_error("unsupported token stream version");
    TokenStream result;
    result.metadata_.token_count = read_value<std::uint64_t>(stream);
    result.metadata_.fingerprint = read_value<std::uint64_t>(stream);
    const auto split_size = read_value<std::uint32_t>(stream);
    result.metadata_.split_id.resize(split_size);
    stream.read(result.metadata_.split_id.data(), split_size);
    if (!stream || result.metadata_.token_count < 2 || result.metadata_.token_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("invalid token stream header");
    }
    result.tokens_.resize(static_cast<std::size_t>(result.metadata_.token_count));
    stream.read(reinterpret_cast<char*>(result.tokens_.data()), static_cast<std::streamsize>(result.tokens_.size() * sizeof(std::int32_t)));
    if (!stream || token_fingerprint(result.tokens_, result.metadata_.split_id) != result.metadata_.fingerprint) {
        throw std::runtime_error("token stream fingerprint mismatch");
    }
    return result;
}

Tensor TokenStream::sample_batch(
    std::int64_t batch_size,
    std::int64_t sequence_length,
    torch::Generator& generator,
    const torch::Device& device) const {
    if (batch_size <= 0 || sequence_length <= 0) throw std::invalid_argument("batch and sequence must be positive");
    const auto maximum_start = static_cast<std::int64_t>(tokens_.size()) - sequence_length;
    if (maximum_start <= 0) throw std::invalid_argument("token stream is too short for requested sequence");
    const auto starts = torch::randint(
        maximum_start,
        {batch_size},
        generator,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    const auto offsets = torch::arange(sequence_length + 1, torch::TensorOptions().dtype(torch::kLong));
    const auto indices = starts.unsqueeze(1) + offsets.unsqueeze(0);
    auto source = torch::from_blob(
        const_cast<std::int32_t*>(tokens_.data()),
        {static_cast<std::int64_t>(tokens_.size())},
        torch::TensorOptions().dtype(torch::kInt32));
    return source.index_select(0, indices.reshape({-1}))
        .reshape({batch_size, sequence_length + 1})
        .to(device, torch::kLong);
}

}  // namespace rayvan::emc
