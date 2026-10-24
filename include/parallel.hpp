#pragma once
#include "mandelbrot.hpp"
#include <cstdint>
#include <vector>

enum class SchedulePolicy { Static, Dynamic8 };
struct ThreadWork { std::uint64_t rows = 0, pixels = 0, iterations = 0; };
struct ParallelResult {
    bool implemented = false;
    int actual_threads = 0;
    std::vector<ThreadWork> work;
};
// The output must be pre-sized. ThreadWork is written independently by each thread.
ParallelResult mandelbrot_parallel(const MandelbrotConfig& cfg,
     std::vector<int>& output, int requested_threads, SchedulePolicy schedule);
