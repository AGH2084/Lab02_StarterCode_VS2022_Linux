#include "parallel.hpp"
#include <stdexcept>

ParallelResult mandelbrot_parallel(const MandelbrotConfig& cfg,
     std::vector<int>& output, int requested_threads, SchedulePolicy schedule) {
    validate_config(cfg);
    if (requested_threads < 1 || requested_threads > 256 ||
        output.size() != static_cast<std::size_t>(cfg.width) * cfg.height)
        throw std::invalid_argument("invalid thread count or output size");
    (void)schedule;
    // TODO 1: Parallelize the independent image rows with OpenMP.
    // TODO 2: Support both static and dynamic,8 scheduling, and collect
    //         per-thread rows/pixels/iterations without shared counter races.
    // Keep the NOT_IMPLEMENTED status until both TODO tasks are completed.
    // Do not substitute the sequential implementation as a parallel solution.
    return ParallelResult{};
}
