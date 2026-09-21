#include "controller.hpp"

#include <iostream>

int main() {
    using namespace smartspray;
    // One synthetic batch: a/b touch on channel 2; c runs independently.
    const std::vector<Target> targets{
        {"b", 1'000'000, 0.625, 1.25},
        {"a", 1'000'000, 0.625, 1.0},
        {"c", 1'000'000, 0.875, 1.0},
        {"outside", 1'000'000, 2.0, 1.0},
        {"late", 1'000'000, 0.0, 0.0}};
    const Config config{0.25, 2.0, 50'000, 125'000};
    const auto result = plan_batch(targets, config, 1'100'000);
    if (const auto* failure = std::get_if<BatchFailure>(&result)) {
        std::cerr << "batch: " << reason_name(failure->reason) << '\n';
        return 1;
    }

    const auto& batch = std::get<BatchPlan>(result);
    std::cout << "plans: target_id nozzle_index arrival_time_us on_time_us off_time_us\n";
    for (const auto& target_result : batch.target_results) {
        if (const auto* pulse = std::get_if<Pulse>(&target_result)) {
            std::cout << pulse->target_id << ' ' << pulse->nozzle_index << ' '
                      << pulse->arrival_time_us << ' ' << pulse->on_time_us << ' '
                      << pulse->off_time_us << '\n';
        } else {
            const auto& rejection = std::get<Rejection>(target_result);
            std::cout << "rejected " << rejection.target_id << ' '
                      << reason_name(rejection.reason) << '\n';
        }
    }
    std::cout << "merged: nozzle_index on_time_us off_time_us source_target_ids\n";
    for (const auto& interval : batch.merged_intervals) {
        std::cout << interval.nozzle_index << ' ' << interval.on_time_us << ' '
                  << interval.off_time_us;
        for (const auto& id : interval.source_target_ids) {
            std::cout << ' ' << id;
        }
        std::cout << '\n';
    }

    VirtualExecutor executor{batch.schedule};
    if (executor.advance_to(1'800'000) != AdvanceResult::OK) {
        std::cerr << "CLOCK_REWIND\n";
        return 1;
    }
    std::cout << "event_time_us nozzle_index command\n";
    for (const auto& event : executor.event_log()) {
        std::cout << event.event_time_us << ' ' << event.nozzle_index << ' '
                  << (event.command == Command::ON ? "ON" : "OFF") << '\n';
    }
    std::cout << "time_us=" << executor.now_us() << " channels[0..7]=";
    for (std::size_t i = 0; i < nozzle_count; ++i) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << (executor.channel_states()[i] == Command::ON ? "ON" : "OFF");
    }
    std::cout << '\n';
}
