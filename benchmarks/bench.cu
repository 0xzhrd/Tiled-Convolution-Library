/*
bench: the --benchmark mode. Times both tiled variants and every baseline over the grid
  filter radius x input size x batch x border mode
and writes one CSV row per (cell, runner). The CSV is the project's headline evidence; the
heatmap and the crossover writeup are drawn from it.

  bench --benchmark [--out FILE] [--warmup N] [--iters N] [--quick] [--label NAME]
  bench --plan      [--quick]          print the grid without touching the GPU

What is timed: only the call that does the convolution, bracketed by CUDA events on the
stream (GPU time, not host time). Setup (allocation, input upload, filter upload, cuDNN
algorithm search, NPP context, OpenCV filter creation) happens before timing.

Every runner gets a row, including the ones that cannot run a cell: "unsupported" (the
library cannot do that border/size), "skipped" (outside our border contract), or "error"
(a failure, with its message). A missing row would hide a finding; a recorded one is data.

Filter coefficients do not affect timing (same FMAs whatever the values), so each radius uses
one fixed-seed random filter. All 15 odd sizes 3x3..31x31 are swept.

Build:
  nvcc -std=c++14 -O3 -I include -I benchmarks/baselines \
    -gencode arch=compute_87,code=sm_87 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 benchmarks/bench.cu -o build/bench \
    -lcudnn -lnppif -lnppc
  (add -DT_CONVL_WITH_OPENCV and the OpenCV libs for the OpenCV column)
*/

#include "t_convl/boundary.hpp"
#include "t_convl/convl_cache.cuh"
#include "t_convl/convl_halo.cuh"
#include "t_convl/convl2d.cuh"
#include "t_convl/device_buffer.hpp"
#include "t_convl/filter_bank.cuh"

#include "common_baseline.hpp"
#include "cudnn_baseline.cuh"
#include "npp_baseline.cuh"
#include "opencv_baseline.cuh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <npp.h>

using namespace tiled_convolution;

namespace
{

// ===========================================================================
// Command line
// ===========================================================================
struct Config
{
    bool        benchmark = false;
    bool        plan      = false;
    bool        quick     = false;
    std::string out       = "bench.csv";
    std::string label     = "";
    int         warmup    = 5;
    int         iters     = 50;
};

void usage()
{
    std::printf(
        "usage: bench --benchmark [--out FILE] [--warmup N] [--iters N] [--quick] [--label NAME]\n"
        "       bench --plan [--quick]\n"
        "  --benchmark  run the sweep and write the CSV\n"
        "  --plan       print the grid and the number of timed runs, without a GPU\n"
        "  --quick      small subset of the grid, for a fast check\n"
        "  --out FILE   CSV path (default bench.csv)\n"
        "  --warmup N   untimed runs per cell before measuring (default 5)\n"
        "  --iters N    timed runs per cell (default 50)\n"
        "  --label NAME free text recorded in the CSV header, e.g. orin-maxn\n");
}

int parse_int(const std::string& flag, const char* text)
{
    std::size_t used = 0;
    int value = 0;
    try
    {
        value = std::stoi(text, &used);
    }
    catch (const std::exception&)
    {
        used = 0;
    }
    if (used == 0 || text[used] != '\0' || value < 1)
        throw std::invalid_argument(flag + " needs a positive integer, got '" + text + "'");
    return value;
}

Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        // Flags that take a value read the next argument; fail clearly if it is missing.
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc)
                throw std::invalid_argument(std::string(flag) + " needs a value");
            return argv[++i];
        };

        if (arg == "--benchmark")   cfg.benchmark = true;
        else if (arg == "--plan")   cfg.plan = true;
        else if (arg == "--quick")  cfg.quick = true;
        else if (arg == "--out")    cfg.out = next("--out");
        else if (arg == "--label")  cfg.label = next("--label");
        else if (arg == "--warmup") cfg.warmup = parse_int("--warmup", next("--warmup"));
        else if (arg == "--iters")  cfg.iters = parse_int("--iters", next("--iters"));
        else if (arg == "--help" || arg == "-h")
        {
            usage();
            std::exit(0);
        }
        else
            throw std::invalid_argument("unknown argument '" + arg + "'");
    }
    if (cfg.benchmark == cfg.plan)
        throw std::invalid_argument("choose exactly one of --benchmark or --plan");
    return cfg;
}

// ===========================================================================
// The grid
// ===========================================================================
struct Size
{
    int w, h;
};

// 16 input sizes: small squares, classic video formats, and the 4K frame Eagle Eye ingests.
const Size kSizes[] = {{64, 64},     {128, 128},   {256, 256},   {320, 240},
                       {512, 512},   {640, 480},   {800, 600},   {960, 540},
                       {1024, 1024}, {1280, 720},  {1600, 900},  {1920, 1080},
                       {2048, 2048}, {2560, 1440}, {3200, 1800}, {3840, 2160}};
const Size kQuickSizes[] = {{256, 256}, {1280, 720}, {1920, 1080}};

const int kBatches[]      = {1, 4, 8};
const int kQuickBatches[] = {1};

const BorderMode kBorders[] = {BorderMode::Zero, BorderMode::Clamp, BorderMode::Reflect101};

std::vector<Size> sizes(const Config& c)
{
    return c.quick ? std::vector<Size>(std::begin(kQuickSizes), std::end(kQuickSizes))
                   : std::vector<Size>(std::begin(kSizes), std::end(kSizes));
}

std::vector<int> batches(const Config& c)
{
    return c.quick ? std::vector<int>(std::begin(kQuickBatches), std::end(kQuickBatches))
                   : std::vector<int>(std::begin(kBatches), std::end(kBatches));
}

std::vector<int> radii(const Config& c)
{
    if (c.quick)
        return {1, 3, 7, 15};
    std::vector<int> r;
    for (int i = 1; i <= max_filter_radius; ++i) // 3x3 .. 31x31
        r.push_back(i);
    return r;
}

// Runner names in CSV order. OpenCV only when built in.
std::vector<std::string> runner_names()
{
    std::vector<std::string> n = {"halo", "cache", "cudnn", "npp"};
#ifdef T_CONVL_WITH_OPENCV
    n.push_back("opencv");
#endif
    return n;
}

// ===========================================================================
// Statistics
// ===========================================================================
struct Stats
{
    double median_ms = 0, p99_ms = 0, min_ms = 0, mean_ms = 0;
};

// Nearest-rank percentile on sorted data: the smallest value with at least p% of samples <= it.
double percentile(const std::vector<float>& sorted, double p)
{
    const std::size_t n    = sorted.size();
    std::size_t       rank = static_cast<std::size_t>(p / 100.0 * static_cast<double>(n) + 0.999999);
    if (rank < 1) rank = 1;
    if (rank > n) rank = n;
    return sorted[rank - 1];
}

Stats summarize(std::vector<float> ms)
{
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.min_ms    = ms.front();
    s.median_ms = percentile(ms, 50.0);
    s.p99_ms    = percentile(ms, 99.0);
    double sum  = 0;
    for (float v : ms)
        sum += v;
    s.mean_ms = sum / static_cast<double>(ms.size());
    return s;
}

// Busy-waits on the GPU for about `cycles` clock ticks. Queued before each timed run so the GPU
// is still busy while the host launches the work: the `start` event is then timestamped just
// before the work begins, instead of while the GPU sits idle waiting for the launch to arrive.
__global__ void spin_kernel(long long cycles)
{
    const long long t0 = clock64();
    while (clock64() - t0 < cycles)
    {
    }
}

// About 100 us at 2 GHz, 200 us on a 1 GHz Orin: longer than the host needs to queue any runner.
constexpr long long kSpinCycles = 200000;

// Warm up, then time each iteration with a pair of events on the stream.
template <typename Run>
Stats time_runner(Run&& run, cudaStream_t stream, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
        run();
    T_CONVL_CHECK_CUDA(cudaStreamSynchronize(stream)); // warm-up finished, and its errors surfaced

    Event start, end;
    std::vector<float> ms;
    ms.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i)
    {
        spin_kernel<<<1, 1, 0, stream>>>(kSpinCycles); // hide host launch latency (see above)
        start.record(stream);
        run();
        end.record(stream);
        ms.push_back(Event::elapsed_ms(start, end)); // waits for `end`
    }
    T_CONVL_CHECK_CUDA(cudaGetLastError());
    return summarize(ms);
}

// ===========================================================================
// CSV
// ===========================================================================
class Csv
{
public:
    explicit Csv(const std::string& path) : file_(path)
    {
        if (!file_)
            throw std::runtime_error("cannot open '" + path + "' for writing");
    }

    void comment(const std::string& line) { file_ << "# " << line << '\n'; }

    void header()
    {
        file_ << "runner,border,radius,ksize,width,height,batch,status,"
                 "median_ms,p99_ms,min_ms,mean_ms,gpix_per_s,smem_bytes,cudnn_algo,detail\n";
        file_.flush();
    }

    struct Row
    {
        std::string runner, border, status = "ok", detail;
        int         radius = 0, width = 0, height = 0, batch = 0;
        Stats       stats;
        bool        timed      = false;
        long long   smem_bytes = -1; // -1 = not applicable
        int         cudnn_algo = -1;
    };

    void write(const Row& r)
    {
        const double pixels = static_cast<double>(r.width) * r.height * r.batch;
        file_ << r.runner << ',' << r.border << ',' << r.radius << ',' << (2 * r.radius + 1) << ','
              << r.width << ',' << r.height << ',' << r.batch << ',' << r.status << ',';
        if (r.timed)
            file_ << r.stats.median_ms << ',' << r.stats.p99_ms << ',' << r.stats.min_ms << ','
                  << r.stats.mean_ms << ',' << pixels / (r.stats.median_ms * 1e-3) / 1e9 << ',';
        else
            file_ << ",,,,,";
        file_ << (r.smem_bytes >= 0 ? std::to_string(r.smem_bytes) : std::string()) << ','
              << (r.cudnn_algo >= 0 ? std::to_string(r.cudnn_algo) : std::string()) << ','
              << clean(r.detail) << '\n';
        file_.flush(); // a crash later in the sweep must not lose rows already measured
    }

private:
    // Free text must not break the CSV: commas and newlines become harmless characters.
    static std::string clean(std::string s)
    {
        for (char& ch : s)
            if (ch == ',' || ch == '\n' || ch == '\r')
                ch = ';';
        return s;
    }

    std::ofstream file_;
};

// Shared-memory bytes per block for our variants: a compile-time value, picked by radius.
template <int R>
long long smem_for(Variant v)
{
    return static_cast<long long>(v == Variant::Halo ? HaloGeometry<R, 16, 16>::smem_bytes
                                                     : CacheGeometry<R, 16, 16>::smem_bytes);
}

long long smem_bytes(int radius, Variant v)
{
    switch (radius)
    {
    case 1:  return smem_for<1>(v);
    case 2:  return smem_for<2>(v);
    case 3:  return smem_for<3>(v);
    case 4:  return smem_for<4>(v);
    case 5:  return smem_for<5>(v);
    case 6:  return smem_for<6>(v);
    case 7:  return smem_for<7>(v);
    case 8:  return smem_for<8>(v);
    case 9:  return smem_for<9>(v);
    case 10: return smem_for<10>(v);
    case 11: return smem_for<11>(v);
    case 12: return smem_for<12>(v);
    case 13: return smem_for<13>(v);
    case 14: return smem_for<14>(v);
    case 15: return smem_for<15>(v);
    default: return -1;
    }
}

// ===========================================================================
// Running one runner on one cell
// ===========================================================================

// A CUDA error that poisons the context (illegal address, launch failure) makes every later
// measurement meaningless. Detect it after any failure and stop the sweep.
void abort_if_context_broken()
{
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess)
        throw std::runtime_error(std::string("CUDA context is broken, stopping the sweep: ") +
                                 cudaGetErrorString(sync));
    cudaGetLastError(); // clear any non-sticky error so the next cell starts clean
}

// Construct (untimed), then time. Unsupported and errors become rows, not crashes.
template <typename Make>
void measure(Csv& csv, Csv::Row row, Make&& make, cudaStream_t stream, const Config& cfg)
{
    try
    {
        auto runner = make(); // setup: never timed
        row.stats   = time_runner([&] { runner->run(); }, stream, cfg.warmup, cfg.iters);
        row.timed   = true;
        row.cudnn_algo = runner->algorithm();
    }
    catch (const baseline::Unsupported& e)
    {
        row.status = "unsupported";
        row.detail = e.what();
    }
    catch (const std::exception& e)
    {
        row.status = "error";
        row.detail = e.what();
        abort_if_context_broken();
    }
    csv.write(row);
}

// Uniform wrappers so measure() can treat every runner the same way.
struct OurVariant
{
    const FilterBank&      bank;
    int                    radius;
    Image<const float> in;
    Image<float>       out;
    cudaStream_t           stream;
    ConvOptions            opt;
    void run() { convl2d(bank, 0, radius, in, out, stream, opt); }
    int  algorithm() const { return -1; }
};

template <typename Lib>
struct Baseline
{
    Lib  lib;
    template <typename... A>
    explicit Baseline(A&&... a) : lib(std::forward<A>(a)...) {}
    void run() { lib.run(); }
    int  algorithm() const { return -1; }
};

template <>
int Baseline<baseline::CudnnConv>::algorithm() const
{
    return lib.algorithm();
}

std::string iso_time()
{
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

void write_environment(Csv& csv, const Config& cfg)
{
    int dev = 0;
    T_CONVL_CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp p;
    T_CONVL_CHECK_CUDA(cudaGetDeviceProperties(&p, dev));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    const NppLibraryVersion* npp = nppGetLibVersion();

    std::ostringstream o;
    csv.comment("tiled convolution benchmark, " + iso_time());
    if (!cfg.label.empty())
        csv.comment("label: " + cfg.label);
    o << "device: " << p.name << " sm_" << p.major << p.minor << ", " << p.multiProcessorCount << " SMs";
    csv.comment(o.str());
    o.str("");
    o << "cuda runtime " << rt << ", driver " << drv << ", cudnn " << cudnnGetVersion() << ", npp "
      << npp->major << '.' << npp->minor << '.' << npp->build;
    csv.comment(o.str());
    o.str("");
    o << "warmup " << cfg.warmup << ", iters " << cfg.iters << ", tile 16x16, input fp32"
      << (cfg.quick ? ", QUICK grid" : "");
    csv.comment(o.str());
}

int plan(const Config& cfg)
{
    const auto s = sizes(cfg);
    const auto b = batches(cfg);
    const auto r = radii(cfg);
    const auto n = runner_names();
    const long long cells = static_cast<long long>(s.size()) * b.size() * r.size() * 3;
    std::printf("grid: %zu sizes x %zu batches x %zu radii x 3 borders = %lld cells\n", s.size(), b.size(),
                r.size(), cells);
    std::printf("runners per cell: %zu (", n.size());
    for (std::size_t i = 0; i < n.size(); ++i)
        std::printf("%s%s", n[i].c_str(), i + 1 < n.size() ? ", " : ")\n");
    std::printf("CSV rows: %lld; timed launches at most %lld (%d warmup + %d iters each)\n",
                cells * static_cast<long long>(n.size()),
                cells * static_cast<long long>(n.size()) * (cfg.warmup + cfg.iters), cfg.warmup, cfg.iters);
    return 0;
}

int run_benchmark(const Config& cfg)
{
    Csv csv(cfg.out);
    write_environment(csv, cfg);
    csv.header();

    Stream       stream;
    FilterBank   bank;
    std::mt19937 rng(20261107u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const auto all_sizes = sizes(cfg);
    const auto all_batch = batches(cfg);
    const auto all_radii = radii(cfg);
    const auto t0        = std::chrono::steady_clock::now();

    for (const Size& sz : all_sizes)
    {
        for (int batch : all_batch)
        {
            // Allocated once per (size, batch) and reused by every radius, border and runner.
            PitchedImage<float> in(sz.w, sz.h, batch), out(sz.w, sz.h, batch);
            std::vector<float>  host(static_cast<std::size_t>(sz.w) * sz.h * batch);
            for (float& p : host)
                p = dist(rng);
            in.upload(host, stream.get());

            for (int r : all_radii)
            {
                const int fw = 2 * r + 1;
                std::vector<float> taps(static_cast<std::size_t>(fw) * fw);
                for (float& t : taps)
                    t = dist(rng);
                const Filter filter("bench_r" + std::to_string(r), r, taps);
                bank.upload(0, filter, stream.get());

                for (BorderMode border : kBorders)
                {
                    Csv::Row base;
                    base.border = border_name(border);
                    base.radius = r;
                    base.width  = sz.w;
                    base.height = sz.h;
                    base.batch  = batch;

                    // Our two variants, through the public API with the variant forced.
                    const Variant variants[] = {Variant::Halo, Variant::Cache};
                    for (Variant v : variants)
                    {
                        Csv::Row row   = base;
                        row.runner     = variant_name(v);
                        row.smem_bytes = smem_bytes(r, v);
                        if (border == BorderMode::Reflect101 && (sz.w <= r || sz.h <= r))
                        {
                            row.status = "skipped";
                            row.detail = "reflect101 needs width and height > radius";
                            csv.write(row);
                            continue;
                        }
                        ConvOptions opt;
                        opt.border           = border;
                        opt.variant          = v;
                        opt.batch            = batch;
                        opt.in_image_stride  = in.image_stride();
                        opt.out_image_stride = out.image_stride();
                        measure(csv, row, [&] {
                            return std::unique_ptr<OurVariant>(new OurVariant{
                                bank, r, in.const_view(), out.view(), stream.get(), opt});
                        }, stream.get(), cfg);
                    }

                    // Baselines: same pixels, same filter, same stream.
                    Csv::Row row = base;
                    row.runner   = "cudnn";
                    measure(csv, row, [&] {
                        return std::unique_ptr<Baseline<baseline::CudnnConv>>(new Baseline<baseline::CudnnConv>(
                            filter, in.const_view(), out.view(), batch, in.image_stride(), out.image_stride(),
                            border, stream.get()));
                    }, stream.get(), cfg);

                    row        = base;
                    row.runner = "npp";
                    measure(csv, row, [&] {
                        return std::unique_ptr<Baseline<baseline::NppConv>>(new Baseline<baseline::NppConv>(
                            filter, in.const_view(), out.view(), batch, in.image_stride(), out.image_stride(),
                            border, stream.get()));
                    }, stream.get(), cfg);

#ifdef T_CONVL_WITH_OPENCV
                    row        = base;
                    row.runner = "opencv";
                    measure(csv, row, [&] {
                        return std::unique_ptr<Baseline<baseline::OpencvConv>>(new Baseline<baseline::OpencvConv>(
                            filter, in.const_view(), out.view(), batch, in.image_stride(), out.image_stride(),
                            border, stream.get()));
                    }, stream.get(), cfg);
#endif
                }
            }

            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "done %dx%d batch %d  (%.0f s elapsed)\n", sz.w, sz.h, batch, elapsed);
        }
    }

    std::fprintf(stderr, "wrote %s\n", cfg.out.c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Config cfg = parse_args(argc, argv);
        if (cfg.plan)
            return plan(cfg);

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        {
            std::fprintf(stderr, "bench: no usable GPU\n");
            return 1;
        }
        return run_benchmark(cfg);
    }
    catch (const std::invalid_argument& e)
    {
        std::fprintf(stderr, "bench: %s\n\n", e.what());
        usage();
        return 2;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "bench: %s\n", e.what());
        return 1;
    }
}
