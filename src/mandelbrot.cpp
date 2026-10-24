#include "mandelbrot.hpp"
#include <numeric>
#include <stdexcept>
#include <cstddef>

std::size_t image_index(int row, int col, int width) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col);
}
int pixel_iterations(int row, int col, const MandelbrotConfig& cfg) {
    const double cr = -2.1 + 3.2 * col / (cfg.width - 1.0);
    const double ci = -1.2 + 2.4 * row / (cfg.height - 1.0);
    double zr = 0.0, zi = 0.0;
    int it = 0;
    while (zr * zr + zi * zi <= 4.0 && it < cfg.max_iterations) {
        const double next = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = next;
        ++it;
    }
    return it;
}
void validate_config(const MandelbrotConfig& cfg) {
    if (cfg.width < 2 || cfg.height < 2 || cfg.max_iterations < 1 ||
        cfg.width > 100000 || cfg.height > 100000 || cfg.max_iterations > 1000000 ||
        static_cast<std::uint64_t>(cfg.width) * cfg.height > 100000000ULL)
        throw std::invalid_argument("width/height must be >=2, iterations >=1, and dimensions within limits");
}
void mandelbrot_sequential(const MandelbrotConfig& cfg, std::vector<int>& output) {
    validate_config(cfg);
    if (output.size() != static_cast<std::size_t>(cfg.width) * cfg.height)
        throw std::invalid_argument("output must be pre-sized to width*height");
    for (int row = 0; row < cfg.height; ++row)
        for (int col = 0; col < cfg.width; ++col)
            output[image_index(row,col,cfg.width)] = pixel_iterations(row,col,cfg);
}
std::uint64_t checksum(const std::vector<int>& output) {
    return std::accumulate(output.begin(), output.end(), std::uint64_t{0});
}
