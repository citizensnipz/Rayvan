#include "rayvan_emc/checkpoint/checkpoint.hpp"

#include <charconv>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace rayvan::emc {
namespace {

std::unordered_map<std::string, std::string> read_manifest(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read checkpoint manifest: " + path.string());
    std::unordered_map<std::string, std::string> fields;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto separator = line.find('=');
        if (separator != std::string::npos) fields.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    if (fields["format"] != "rayvan-emc-checkpoint-v1") throw std::runtime_error("unsupported checkpoint format");
    return fields;
}

template <typename T>
T number(const std::unordered_map<std::string, std::string>& fields, const std::string& key) {
    const auto iterator = fields.find(key);
    if (iterator == fields.end()) throw std::runtime_error("checkpoint manifest missing: " + key);
    if constexpr (std::is_floating_point_v<T>) {
        std::size_t consumed = 0;
        const auto value = static_cast<T>(std::stod(iterator->second, &consumed));
        if (consumed != iterator->second.size()) throw std::runtime_error("invalid checkpoint number: " + key);
        return value;
    } else {
        T value{};
        const auto* begin = iterator->second.data();
        const auto* end = begin + iterator->second.size();
        if (const auto result = std::from_chars(begin, end, value); result.ec != std::errc{} || result.ptr != end) {
            throw std::runtime_error("invalid checkpoint number: " + key);
        }
        return value;
    }
}

CheckpointProgress progress_from(const std::unordered_map<std::string, std::string>& fields) {
    CheckpointProgress progress;
    progress.step = number<std::int64_t>(fields, "step");
    progress.tokens_processed = number<std::int64_t>(fields, "tokens_processed");
    progress.validation_loss = number<double>(fields, "validation_loss");
    progress.best_validation_loss = number<double>(fields, "best_validation_loss");
    progress.seed = number<std::uint64_t>(fields, "seed");
    progress.precision = fields.at("precision");
    return progress;
}

void replace_file(const std::filesystem::path& temporary, const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::remove(destination, error);
    error.clear();
    std::filesystem::rename(temporary, destination, error);
    if (error) throw std::runtime_error("cannot finalize checkpoint file: " + destination.string());
}

}  // namespace

void save_checkpoint(
    const std::filesystem::path& directory,
    const EMCModel& model,
    const torch::optim::Optimizer* optimizer,
    const CheckpointProgress& progress) {
    std::filesystem::create_directories(directory);
    save_model_config(model.config(), directory / "model.rvcfg");

    const auto model_temporary = directory / "model.pt.tmp";
    model.save_weights(model_temporary);
    replace_file(model_temporary, directory / "model.pt");
    if (optimizer) {
        torch::serialize::OutputArchive archive;
        optimizer->save(archive);
        const auto temporary = directory / "optimizer.pt.tmp";
        archive.save_to(temporary.string());
        replace_file(temporary, directory / "optimizer.pt");
    }
    const bool has_rng =
        progress.cpu_rng_state.has_value() ||
        progress.train_generator_state.has_value() ||
        progress.evaluation_generator_state.has_value() ||
        progress.cuda_rng_state.has_value();
    if (has_rng) {
        torch::serialize::OutputArchive archive;
        if (progress.cpu_rng_state) {
            archive.write("cpu_rng_state", *progress.cpu_rng_state, true);
        }
        if (progress.train_generator_state) {
            archive.write(
                "train_generator_state",
                *progress.train_generator_state,
                true);
        }
        if (progress.evaluation_generator_state) {
            archive.write(
                "evaluation_generator_state",
                *progress.evaluation_generator_state,
                true);
        }
        if (progress.cuda_rng_state) {
            archive.write("cuda_rng_state", *progress.cuda_rng_state, true);
        }
        const auto temporary = directory / "rng.pt.tmp";
        archive.save_to(temporary.string());
        replace_file(temporary, directory / "rng.pt");
    }

    const auto manifest_temporary = directory / "manifest.rayvan.tmp";
    std::ofstream manifest(manifest_temporary, std::ios::binary | std::ios::trunc);
    if (!manifest) throw std::runtime_error("cannot write checkpoint manifest");
    manifest << "format=rayvan-emc-checkpoint-v1\n"
             << "format_version=" << checkpoint_format_version << '\n'
             << "model_file=model.pt\n"
             << "optimizer_file=" << (optimizer ? "optimizer.pt" : "") << '\n'
             << "config_file=model.rvcfg\n"
             << "rng_file=" << (has_rng ? "rng.pt" : "") << '\n'
             << "active_top_k=" << model.active_top_k() << '\n'
             << "step=" << progress.step << '\n'
             << "tokens_processed=" << progress.tokens_processed << '\n'
             << "validation_loss=" << progress.validation_loss << '\n'
             << "best_validation_loss=" << progress.best_validation_loss << '\n'
             << "seed=" << progress.seed << '\n'
             << "precision=" << progress.precision << '\n';
    manifest.close();
    replace_file(manifest_temporary, directory / "manifest.rayvan");
}

CheckpointProgress load_model_checkpoint(
    const std::filesystem::path& directory,
    EMCModel& model,
    const torch::Device& device) {
    const auto fields = read_manifest(directory / "manifest.rayvan");
    const auto stored_config = load_model_config(directory / fields.at("config_file"));
    if (stored_config.n1_mode != model.config().n1_mode ||
        stored_config.population != model.config().population || stored_config.latent_dim != model.config().latent_dim ||
        stored_config.vocab_size != model.config().vocab_size || stored_config.n1_depth != model.config().n1_depth) {
        throw std::runtime_error("checkpoint architecture does not match model");
    }
    model.load_weights(directory / fields.at("model_file"), device);
    if (model.config().n1_mode == N1Mode::legacy_nexus) {
        model.set_active_top_k(number<std::int64_t>(fields, "active_top_k"));
    }
    return progress_from(fields);
}

CheckpointProgress load_training_checkpoint(
    const std::filesystem::path& directory,
    EMCModel& model,
    torch::optim::Optimizer& optimizer,
    const torch::Device& device) {
    auto progress = load_model_checkpoint(directory, model, device);
    const auto fields = read_manifest(directory / "manifest.rayvan");
    const auto optimizer_file = fields.at("optimizer_file");
    if (optimizer_file.empty()) throw std::runtime_error("checkpoint has no optimizer state");
    torch::serialize::InputArchive archive;
    archive.load_from((directory / optimizer_file).string(), device);
    optimizer.load(archive);
    const auto rng_iterator = fields.find("rng_file");
    const auto rng_file = rng_iterator == fields.end()
        ? std::string()
        : rng_iterator->second;
    if (!rng_file.empty()) {
        torch::serialize::InputArchive rng_archive;
        rng_archive.load_from((directory / rng_file).string(), torch::kCPU);
        Tensor state;
        if (rng_archive.try_read("cpu_rng_state", state, true)) {
            progress.cpu_rng_state = state;
        }
        if (rng_archive.try_read("train_generator_state", state, true)) {
            progress.train_generator_state = state;
        }
        if (rng_archive.try_read(
                "evaluation_generator_state", state, true)) {
            progress.evaluation_generator_state = state;
        }
        if (rng_archive.try_read("cuda_rng_state", state, true)) {
            progress.cuda_rng_state = state;
        }
    }
    return progress;
}

std::filesystem::path milestone_checkpoint_path(
    const std::filesystem::path& root,
    std::int64_t tokens_processed) {
    return root / ("tokens-" + std::to_string(tokens_processed));
}

}  // namespace rayvan::emc
