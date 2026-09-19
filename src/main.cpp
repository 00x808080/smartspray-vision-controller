#include "controller.hpp"

#include <iostream>

int main() {
    using namespace smartspray;
    const Target target{"synthetic-1", 1'000'000, 0.625, 1.0};
    const Config config{0.25, 2.0, 50'000, 100'000};
    const auto result = plan_pulse(target, config, 1'100'000);
    if (const auto* rejection = std::get_if<Rejection>(&result)) {
        std::cerr << rejection->target_id << ": " << reason_name(rejection->reason) << '\n';
        return 1;
    }

    const auto& pulse = std::get<Pulse>(result);
    std::cout << "target_id=" << pulse.target_id << '\n'
              << "nozzle_index=" << pulse.nozzle_index << '\n'
              << "arrival_time_us=" << pulse.arrival_time_us << '\n'
              << "on_time_us=" << pulse.on_time_us << '\n'
              << "off_time_us=" << pulse.off_time_us << '\n';

    VirtualExecutor executor{pulse};
    if (executor.advance_to(1'600'000) != AdvanceResult::OK) {
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
