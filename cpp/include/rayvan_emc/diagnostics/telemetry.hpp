#pragma once

#include "rayvan_emc/diagnostics/diagnostics.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>

namespace rayvan::emc {

struct TelemetryRecord {
    std::int64_t step = 0;
    std::int64_t tokens_processed = 0;
    double loss = 0.0;
    double perplexity = 0.0;
    double tokens_per_second = 0.0;
    double wall_seconds = 0.0;
    double parameter_norm = 0.0;
    double gradient_norm = 0.0;
    double update_norm = 0.0;
    RoutingReport routing;
    IntegratorReport integrator;
    std::optional<RoutingFreeReport> routing_free;
    MemoryReport memory;
};

// Appends one tab-separated record. Large device-side accumulators are only
// materialized when this milestone function is called.
void append_telemetry(const std::filesystem::path& path, const TelemetryRecord& record);

}  // namespace rayvan::emc
