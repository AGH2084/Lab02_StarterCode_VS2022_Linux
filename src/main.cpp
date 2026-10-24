#include "mandelbrot.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

struct Options {
    std::string mode = "profile";
    MandelbrotConfig cfg;
    int passes = 8, repeats = 5;
    std::vector<int> threads{1,2,4,8};
};
static int positive(const std::string& text, int maximum = 1000000) {
    std::size_t n = 0;
    const long long value = std::stoll(text, &n);
    if (n != text.size() || value < 1 || value > maximum)
        throw std::invalid_argument("invalid positive integer: " + text);
    return static_cast<int>(value);
}
static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "Usage: lab02 [--mode profile|self-test|experiment] [--width N] [--height N]"
                         " [--iterations N] [--passes N] [--repeats N] [--threads 1,2,4,8]\n"
                         "Default mode: profile. Use x64 Release for benchmarks.\n";
            std::exit(0);
        }
        if (i+1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--mode") o.mode = value;
        else if (key == "--width") o.cfg.width = positive(value);
        else if (key == "--height") o.cfg.height = positive(value);
        else if (key == "--iterations") o.cfg.max_iterations = positive(value);
        else if (key == "--passes") o.passes = positive(value, 1000);
        else if (key == "--repeats") o.repeats = positive(value, 100);
        else if (key == "--threads") {
            o.threads.clear();
            std::size_t start=0;
            do {
                auto end = value.find(',',start);
                const auto token = value.substr(start, end==std::string::npos ? end : end-start);
                o.threads.push_back(positive(token,256));
                if (end==std::string::npos) break;
                start=end+1;
            } while (start < value.size());
            if (value.empty() || value.back()==',') throw std::invalid_argument("invalid thread list");
        } else throw std::invalid_argument("unknown option: " + key);
    }
    validate_config(o.cfg);
    if (o.mode!="profile" && o.mode!="self-test" && o.mode!="experiment")
        throw std::invalid_argument("unknown mode: " + o.mode);
    return o;
}
static double timed_sequential(const MandelbrotConfig& cfg, std::vector<int>& pixels) {
    const auto t0 = std::chrono::steady_clock::now();
    mandelbrot_sequential(cfg,pixels);
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}
static double timed_parallel(const MandelbrotConfig& cfg, std::vector<int>& pixels,
                             int threads, SchedulePolicy schedule, ParallelResult& result) {
    const auto t0 = std::chrono::steady_clock::now();
    result = mandelbrot_parallel(cfg,pixels,threads,schedule);
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}
static double median(std::vector<double> a) {
    std::sort(a.begin(),a.end());
    const auto mid=a.size()/2;
    return a.size()%2 ? a[mid] : (a[mid-1]+a[mid])/2.0;
}
static bool verify(const MandelbrotConfig& cfg, const std::vector<int>& reference,
                   const std::vector<int>& observed, int requested, const ParallelResult& r,
                   bool verbose) {
    if (!r.implemented) { if(verbose) std::cout << "NOT_IMPLEMENTED\n"; return false; }
    if (r.actual_threads != requested || r.work.size() != static_cast<std::size_t>(r.actual_threads)) {
        if(verbose) std::cout << "FAIL: actual OpenMP team does not match requested threads\n";
        return false;
    }
    auto mismatch=std::mismatch(reference.begin(),reference.end(),observed.begin());
    if (mismatch.first != reference.end()) {
        if(verbose) std::cout << "FAIL: first mismatch at index " << (mismatch.first-reference.begin()) << '\n';
        return false;
    }
    std::uint64_t rows=0,pixels=0,iterations=0;
    for (const auto& x:r.work) { rows+=x.rows; pixels+=x.pixels; iterations+=x.iterations; }
    if (rows!=static_cast<std::uint64_t>(cfg.height) ||
        pixels!=reference.size() || iterations!=checksum(reference)) {
        if(verbose) std::cout << "FAIL: per-thread statistics violate work invariants\n";
        return false;
    }
    return true;
}
static int profile(const Options& o) {
    std::vector<int> pixels(static_cast<std::size_t>(o.cfg.width)*o.cfg.height);
    std::cout << "mode,pass,seconds,checksum,status\n";
    std::uint64_t expected=0;
    for(int pass=1;pass<=o.passes;++pass) {
        const auto seconds=timed_sequential(o.cfg,pixels);
        const auto sum=checksum(pixels);
        if(pass==1) expected=sum;
        const bool ok=sum==expected;
        std::cout << "profile," << pass << ',' << seconds << ',' << sum << ',' << (ok?"PASS":"FAIL") << '\n';
        if(!ok) return 3;
    }
    return 0;
}
static int self_test(const Options& o) {
    (void)o;
    const std::vector<MandelbrotConfig> cases{{2,2,1},{7,5,15},{11,3,30},
        {3,11,35},{17,19,80},{123,41,100}};
    const std::vector<int> teams{1,2,3,4,8};
    bool all_ok=true, not_implemented=false;
    for(const auto& cfg:cases) {
        std::vector<int> expected(static_cast<std::size_t>(cfg.width)*cfg.height);
        std::vector<int> output(expected.size(),-1);
        mandelbrot_sequential(cfg,expected);
        for(const int threads:teams) for(const auto policy:{SchedulePolicy::Static,SchedulePolicy::Dynamic8}) {
            std::fill(output.begin(),output.end(),-1);
            auto r=mandelbrot_parallel(cfg,output,threads,policy);
            if(!r.implemented) not_implemented=true;
            const bool ok=verify(cfg,expected,output,threads,r,false);
            all_ok &= ok;
            std::cout << "test," << cfg.width << 'x' << cfg.height << ',' << threads << ','
                      << (policy==SchedulePolicy::Static?"static":"dynamic8") << ','
                      << (!r.implemented?"NOT_IMPLEMENTED":(ok?"PASS":"FAIL")) << '\n';
        }
    }
    std::cout << "SELF_TEST," << (not_implemented?"NOT_IMPLEMENTED":(all_ok?"PASS":"FAIL")) << '\n';
    return not_implemented?4:(all_ok?0:3);
}
static int experiment(const Options& o) {
    const auto n=static_cast<std::size_t>(o.cfg.width)*o.cfg.height;
    std::vector<int> expected(n),output(n,-1);
    mandelbrot_sequential(o.cfg,expected); // warm-up, also prepares the reference
    std::vector<double> baseline;
    for(int i=0;i<o.repeats;++i) baseline.push_back(timed_sequential(o.cfg,expected));
    const double base=median(baseline);
    std::cout << "sequential_median_seconds," << base << ",checksum," << checksum(expected) << '\n';
    std::cout << "threads,schedule,actual_threads,median_seconds,speedup,status\n";
    bool all_ok=true,missing=false;
    for(const int threads:o.threads) for(const auto policy:{SchedulePolicy::Static,SchedulePolicy::Dynamic8}) {
        std::vector<double> times;
        ParallelResult r;
        // Warm-up is excluded from measurements. All runs are validated.
        for(int rep=-1;rep<o.repeats;++rep) {
            std::fill(output.begin(),output.end(),-1);
            const auto seconds=timed_parallel(o.cfg,output,threads,policy,r);
            if(!r.implemented) { missing=true; break; }
            const bool ok=verify(o.cfg,expected,output,threads,r,true);
            if(!ok) { all_ok=false; break; }
            if(rep>=0) times.push_back(seconds);
        }
        const bool ok=times.size()==static_cast<std::size_t>(o.repeats);
        const auto name=policy==SchedulePolicy::Static?"static":"dynamic8";
        std::cout << threads << ',' << name << ',' << r.actual_threads << ',';
        if(ok) std::cout << median(times) << ',' << base/median(times);
        else std::cout << "NA,NA";
        std::cout << ',' << (!r.implemented?"NOT_IMPLEMENTED":(ok?"PASS":"FAIL")) << '\n';
        if(ok) {
            std::cout << "work_distribution,threads,schedule,thread_id,rows,pixels,iterations\n";
            for(std::size_t id=0;id<r.work.size();++id) {
                const auto& w=r.work[id];
                std::cout << "work," << threads << ',' << name << ',' << id << ',' << w.rows << ',' << w.pixels << ',' << w.iterations << '\n';
            }
        }
    }
    return missing?4:(all_ok?0:3);
}
int main(int argc,char** argv) {
    try {
        const auto o=parse(argc,argv);
        std::cout << std::fixed << std::setprecision(6);
        if(o.mode=="profile") return profile(o);
        if(o.mode=="self-test") return self_test(o);
        return experiment(o);
    } catch(const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 2; }
}
