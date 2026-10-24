#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

struct MandelbrotConfig { int width = 1200; int height = 800; int max_iterations = 600; };
std::size_t image_index(int row, int col, int width);
int pixel_iterations(int row, int col, const MandelbrotConfig& cfg);
void mandelbrot_sequential(const MandelbrotConfig& cfg, std::vector<int>& output);
std::uint64_t checksum(const std::vector<int>& output);
void validate_config(const MandelbrotConfig& cfg);
