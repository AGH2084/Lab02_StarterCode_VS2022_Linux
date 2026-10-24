#include "parallel.hpp"
#include <stdexcept>
#include <omp.h>

ParallelResult mandelbrot_parallel(const MandelbrotConfig& cfg,
     std::vector<int>& output, int requested_threads, SchedulePolicy schedule) {
    validate_config(cfg);
    if (requested_threads < 1 || requested_threads > 256 ||
        output.size() != static_cast<std::size_t>(cfg.width) * cfg.height)
        throw std::invalid_argument("invalid thread count or output size");
    ParallelResult result;
    // The team can never exceed the request, so one slot per requested thread is
    // enough; this avoids allocating inside the parallel region.
    result.work.assign(static_cast<std::size_t>(requested_threads), ThreadWork{});
    int team = 0;
    #pragma omp parallel num_threads(requested_threads)
    {
        #pragma omp single
        team = omp_get_num_threads(); // implicit barrier publishes the value
        // Thread-local counters: no sharing (or false sharing) in the hot loop.
        ThreadWork local;
        if (schedule == SchedulePolicy::Static) {
            #pragma omp for schedule(static)
            for (int row = 0; row < cfg.height; ++row) {
                for (int col = 0; col < cfg.width; ++col) {
                    const int it = pixel_iterations(row, col, cfg);
                    output[image_index(row, col, cfg.width)] = it;
                    local.iterations += static_cast<std::uint64_t>(it);
                }
                ++local.rows;
                local.pixels += static_cast<std::uint64_t>(cfg.width);
            }
        } else {
            #pragma omp for schedule(dynamic, 8)
            for (int row = 0; row < cfg.height; ++row) {
                for (int col = 0; col < cfg.width; ++col) {
                    const int it = pixel_iterations(row, col, cfg);
                    output[image_index(row, col, cfg.width)] = it;
                    local.iterations += static_cast<std::uint64_t>(it);
                }
                ++local.rows;
                local.pixels += static_cast<std::uint64_t>(cfg.width);
            }
        }
        // Each thread owns exactly one distinct slot, so this write is race-free.
        result.work[static_cast<std::size_t>(omp_get_thread_num())] = local;
    }
    result.actual_threads = team;
    result.work.resize(static_cast<std::size_t>(team));
    result.implemented = true;
    return result;
}
