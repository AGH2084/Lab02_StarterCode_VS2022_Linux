#include "parallel.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
int main() {
    try {
        const std::vector<MandelbrotConfig> cases{{2,2,1},{7,5,12},{11,3,40},{3,11,25},{17,19,75},{123,41,90}};
        const std::vector<int> teams{1,2,3,4,8};
        bool missing=false,failed=false;
        for(const auto& cfg:cases) {
            validate_config(cfg);
            std::vector<int> reference(static_cast<std::size_t>(cfg.width)*cfg.height), output(reference.size(),-1);
            mandelbrot_sequential(cfg,reference);
            for(int threads:teams) for(auto policy:{SchedulePolicy::Static,SchedulePolicy::Dynamic8}) {
                std::fill(output.begin(),output.end(),-1);
                auto r=mandelbrot_parallel(cfg,output,threads,policy);
                if(!r.implemented) {missing=true;continue;}
                std::uint64_t rows=0,pixels=0,iterations=0;
                for(const auto& w:r.work) {rows+=w.rows; pixels+=w.pixels; iterations+=w.iterations;}
                const bool ok=r.actual_threads==threads && r.work.size()==static_cast<std::size_t>(threads)
                    && reference==output && rows==static_cast<std::uint64_t>(cfg.height)
                    && pixels==reference.size() && iterations==checksum(reference);
                if(!ok) {failed=true;std::cerr << "FAIL for " << cfg.width << 'x' << cfg.height << " threads=" << threads << '\n';}
            }
        }
        if(missing) {std::cout << "NOT_IMPLEMENTED: complete TODO 1 and TODO 2\n";return 4;}
        if(failed) return 3;
        std::cout << "PASS: pixel values, thread teams, and work invariants\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 2;}
}
