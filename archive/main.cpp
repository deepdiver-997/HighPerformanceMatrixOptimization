// #include <xmmintrin.h>
#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <simd/simd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <functional>
#include <atomic>

bool comp(float *a, float *b, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(a[i] - b[i]) > 1e-3) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

struct timer {
    std::chrono::steady_clock::time_point start, end;
    timer() { start = std::chrono::steady_clock::now(); }
    ~timer() {
        // end = std::chrono::steady_clock::now();
        // std::cout << "Elapsed time: " 
        //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
        //           << " ms" << std::endl;
    }
    void start_timer() { start = std::chrono::steady_clock::now(); }
    void end_timer() { 
        end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << " ms" << std::endl;
    }
};

float vec_dot(const float* x, const float* y, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

float* matrix_mul_trans(const float* A, const float* B, int r, int k, int c) {
    float *b = new float[c * k];
    float *res = new float[r * c];
    for (int i = 0; i < c; ++i) {
        for(int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i * c + j] = vec_dot(&A[i * k], &b[j * k], k);
        }
    }
    
    delete[] b;
    return res;
}

void base_mul(float* C, const float* A, const float* B, int r, int k, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float acc = 0.0;
            for (int p = 0; p < k; ++p) {
                acc += A[i * k + p] * B[p * c + j];
            }
            C[i * c + j] = acc;
        }
    }
}

float* matrix_mul_block(const float* A, const float* B, int r, int k, int c, int block_size = 128) {
    float *res = new float[r * c];
    // initialize result to zero to avoid UB from reading uninitialized values
    for (int i = 0; i < r * c; ++i) res[i] = 0.0f;

    for (int i = 0; i < r; i += block_size) {
        for (int j = 0; j < c; j += block_size) {
            for (int p = 0; p < k; p += block_size) {
                int i_end = std::min(i + block_size, r);
                int j_end = std::min(j + block_size, c);
                int p_end = std::min(p + block_size, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * B[pp * c + jj];
                        }
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    return res;
}

float* matrix_mul_trans_block(const float* A, const float* B, int r, int k, int c, int block_size = 128) {
    float *b = new float[c * k];
    float *res = new float[r * c];
    // initialize result to zero to avoid UB from reading uninitialized values
    for (int i = 0; i < r * c; ++i) res[i] = 0.0f;

    for (int i = 0; i < c; ++i) {
        for(int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    for (int i = 0; i < r; i += block_size) {
        for (int j = 0; j < c; j += block_size) {
            for (int p = 0; p < k; p += block_size) {
                int i_end = std::min(i + block_size, r);
                int j_end = std::min(j + block_size, c);
                int p_end = std::min(p + block_size, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        double sum = 0.0;
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += static_cast<double>(A[ii * k + pp]) * static_cast<double>(b[jj * k + pp]);
                        }
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    
    delete[] b;
    return res;
}

// aligned allocation helper: align to 'align' bytes (must be power of two)
static void* aligned_alloc_helper(size_t align, size_t size) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return aligned_alloc(align, ((size + align - 1) / align) * align);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, ((size + align - 1) / align) * align) != 0) return nullptr;
    return ptr;
#endif
}

float* matrix_mul_trans_block_with_neon(const float* A, const float* B, int r, int k, int c, int block_size = 128) {
    const size_t align = 64;
    float *b = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
    float *res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!b || !res) {
        std::cerr << "Aligned allocation failed" << std::endl;
        free(b);
        free(res);
        return nullptr;
    }
    // initialize result to zero to avoid UB from reading uninitialized values
    memset(res, 0, sizeof(float) * r * c);
    for (int i = 0; i < c; ++i) {
        for(int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    for (int i = 0; i < r; i += block_size) {
        for (int j = 0; j < c; j += block_size) {
            for (int p = 0; p < k; p += block_size) {
                int i_end = std::min(i + block_size, r);
                int j_end = std::min(j + block_size, c);
                int p_end = std::min(p + block_size, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        int pp;
                        float32x4_t sum_vec = vdupq_n_f32(0.0f);
                        for (pp = p; pp < p_end; pp+=8) {
                            float32x4_t a1vec = vld1q_f32(&A[ii * k + pp]);
                            float32x4_t a2vec = vld1q_f32(&A[ii * k + pp + 4]);
                            float32x4_t b1_vec = vld1q_f32(&b[jj * k + pp]);
                            float32x4_t b2_vec = vld1q_f32(&b[jj * k + pp + 4]);
                            // accumulate products for 8 elements in an interleaved vector form
                            sum_vec = vaddq_f32(sum_vec, vmlaq_f32(vmulq_f32(a1vec, b1_vec), a2vec, b2_vec));
                        }
                        for (; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * b[jj * k + pp];
                        }
                        // horizontal add of sum_vec: use vaddvq_f32 if available, otherwise manual reduction
#if defined(__ARM_FEATURE_SIMD32) || defined(__aarch64__)
                        float vec_sum = vaddvq_f32(sum_vec);
#else
                        // manual horizontal add
                        float32x2_t lo = vget_low_f32(sum_vec);
                        float32x2_t hi = vget_high_f32(sum_vec);
                        float32x2_t t = vadd_f32(lo, hi);
                        float vec_sum = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
#endif
                        res[ii * c + jj] += static_cast<float>(sum + static_cast<double>(vec_sum));
                    }
                }
            }
        }
    }
    
    delete[] b;
    return res;
}

// Work-stealing thread pool: per-thread queues, submit assigns to a queue, workers steal from others
class WorkStealingThreadPool {
public:
    using Task = std::function<void(int)>; // task receives executing worker id

    WorkStealingThreadPool(size_t num_workers)
    : nworkers(std::max<size_t>(1, num_workers)),
      queues(nworkers),
      qmutex(nworkers),
      next_worker(0),
      local_pending(0),
      submitted(0),
      completed(0),
      stop_flag(false)
    {
        workers.reserve(nworkers);
        for (size_t i = 0; i < nworkers; ++i) {
            workers.emplace_back([this, i]{ this->worker_loop(i); });
        }
    }

    ~WorkStealingThreadPool() {
        {
            std::unique_lock<std::mutex> lk(global_mutex);
            stop_flag = true;
        }
        cv.notify_all();
        for (auto &t : workers) if (t.joinable()) t.join();
    }

    // submit to a queue (round-robin)
    void submit(Task task) {
        size_t idx = next_worker.fetch_add(1) % nworkers;
        {
            std::unique_lock<std::mutex> lk(qmutex[idx]);
            queues[idx].emplace_front(std::move(task));
        }
        ++submitted;
        local_pending.fetch_add(1);
        cv.notify_one();
    }

    // main thread tries to pop a task from any queue (front)
    bool try_pop_task(Task &out_task, int &out_worker_id) {
        for (size_t i = 0; i < nworkers; ++i) {
            std::unique_lock<std::mutex> lk(qmutex[i]);
            if (!queues[i].empty()) {
                out_task = std::move(queues[i].front());
                queues[i].pop_front();
                out_worker_id = static_cast<int>(i);
                local_pending.fetch_sub(1);
                return true;
            }
        }
        return false;
    }

    size_t pending() { return local_pending.load(); }

    void notify_task_completed() {
        auto prev = ++completed;
        if (prev >= submitted.load()) {
            std::unique_lock<std::mutex> lk(done_mutex);
            done_cv.notify_all();
        }
    }

    void wait_for_all() {
        std::unique_lock<std::mutex> lk(done_mutex);
        done_cv.wait(lk, [this]{ return completed.load() >= submitted.load(); });
    }

private:
    void worker_loop(int id) {
        while (true) {
            Task task;
            bool got = false;
            // try pop from own queue
            {
                std::unique_lock<std::mutex> lk(qmutex[id]);
                if (!queues[id].empty()) {
                    task = std::move(queues[id].front());
                    queues[id].pop_front();
                    got = true;
                    local_pending.fetch_sub(1);
                }
            }
            // try steal if didn't get
            if (!got) {
                for (size_t j = 0; j < nworkers; ++j) {
                    size_t k = (id + 1 + j) % nworkers;
                    std::unique_lock<std::mutex> lk(qmutex[k]);
                    if (!queues[k].empty()) {
                        task = std::move(queues[k].back());
                        queues[k].pop_back();
                        got = true;
                        local_pending.fetch_sub(1);
                        break;
                    }
                }
            }

            if (got) {
                task(static_cast<int>(id));
                notify_task_completed();
                continue;
            }

            // nothing to do, wait
            std::unique_lock<std::mutex> lk(global_mutex);
            cv.wait(lk, [this]{ return stop_flag || local_pending.load() > 0; });
            if (stop_flag && local_pending.load() == 0) return;
        }
    }

    size_t nworkers;
    std::vector<std::thread> workers;
    std::vector<std::deque<Task>> queues;
    std::vector<std::mutex> qmutex;
    std::atomic<size_t> next_worker;
    std::atomic<size_t> local_pending;
    std::atomic<size_t> submitted;
    std::atomic<size_t> completed;
    std::mutex global_mutex;
    std::condition_variable cv;
    bool stop_flag;
    std::mutex done_mutex;
    std::condition_variable done_cv;
};

// Parallel block multiplication using thread pool
float* matrix_mul_block_parallel(const float* A, const float* B, int r, int k, int c,
                                 int block_size = 32, int num_workers = 0, size_t queue_threshold = 64) {
    if (num_workers <= 0) num_workers = std::max(1u, std::thread::hardware_concurrency());
    size_t nw = static_cast<size_t>(num_workers);

    // allocate per-thread private result matrices (include main thread index at the end)
    size_t n_priv = nw + 1; // workers + main
    std::vector<float*> priv(n_priv, nullptr);
    for (size_t t = 0; t < n_priv; ++t) {
        priv[t] = new float[r * c];
        std::fill_n(priv[t], r * c, 0.0f);
    }

    WorkStealingThreadPool pool(nw);

    // submit tasks: task signature void(int worker_id) -> write into priv[worker_id]
    for (int i = 0; i < r; i += block_size) {
        int i_end = std::min(i + block_size, r);
        for (int j = 0; j < c; j += block_size) {
            int j_end = std::min(j + block_size, c);

            auto task = [=, &A, &B, &priv](int worker_id) {
                float* out = priv[worker_id];
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        for (int p = 0; p < k; ++p) {
                            sum += A[ii * k + p] * B[p * c + jj];
                        }
                        out[ii * c + jj] = sum;
                    }
                }
            };

            if (pool.pending() > queue_threshold) {
                // execute in main thread; main index is nw
                task(static_cast<int>(nw));
                pool.notify_task_completed();
            } else {
                pool.submit(task);
            }
        }
    }

    // main thread helps by stealing remaining tasks; when executing uses main index nw
    while (true) {
        WorkStealingThreadPool::Task t;
        int owner = -1;
        if (!pool.try_pop_task(t, owner)) break;
        t(static_cast<int>(nw));
        pool.notify_task_completed();
    }

    pool.wait_for_all();

    // merge private buffers into single result. If large, merge in parallel.
    float* res = new float[r * c];
    std::fill_n(res, r * c, 0.0f);
    const size_t total = static_cast<size_t>(r) * static_cast<size_t>(c);
    const size_t parallel_merge_threshold = 1 << 20; // ~1M elements
    if (total >= parallel_merge_threshold && nw > 1) {
        // parallel merge using nw threads
        std::vector<std::thread> mers;
        size_t chunk = (total + nw - 1) / nw;
        for (size_t t = 0; t < nw; ++t) {
            size_t start = t * chunk;
            size_t end = std::min(total, start + chunk);
            mers.emplace_back([start, end, &res, &priv, n_priv]() {
                for (size_t idx = start; idx < end; ++idx) {
                    float s = 0.0f;
                    for (size_t p = 0; p < n_priv; ++p) s += priv[p][idx];
                    res[idx] = s;
                }
            });
        }
        for (auto &th : mers) if (th.joinable()) th.join();
    } else {
        for (size_t idx = 0; idx < total; ++idx) {
            float s = 0.0f;
            for (size_t p = 0; p < n_priv; ++p) s += priv[p][idx];
            res[idx] = s;
        }
    }

    for (size_t t = 0; t < n_priv; ++t) {
        delete[] priv[t];
    }

    return res;
}

float rand_float(float s) {
    return 4 * s * (1 - s);
}

float* random_matrix(int r, int c, float seed) {
    const size_t align = 64;
    float *res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!res) return nullptr;
    for (int i = 0; i < r * c; ++i) {
        res[i] = rand_float(seed);
    }
    return res;
}

float Trace(const float* A, int r, int c) {
    float sum = 0.0f;
    int n = std::min(r, c);
    for (int i = 0; i < n; ++i) {
        sum += A[i * c + i];
    }
    return sum;
}

inline void print_usage() {
    std::cout << "Usage: ./main -s [seed] [size-1] [size-2] ..." << std::endl;
}

int main(int argc, char** argv) {
    int size = 0;
    float seed = 0.3f;
    if (argc == 1) {
        print_usage();
        return 0;
    }
    if (argv[1][0] == '-' && argv[1][1] == 's') {
        if (argc < 4) {
            print_usage();
            return 0;
        }
        seed = atof(argv[2]);
        argc -= 2;
        argv += 2;
    } else {
        print_usage();
        return 0;
    }
    float *p_base;
    while (--argc)
    {
        size = atoi(*(++argv));
        if (size <= 0) {
            print_usage();
            std::cout << "Invalid size input: " << *argv << std::endl;
            return 0;
        }
        auto A = random_matrix(size, size, seed);
        auto B = random_matrix(size, size, seed);
        std::cout << "Matrix multiplication performance comparison (" << size << " x " << size << "matrices):" << std::endl;

        {
            std::cout << "Base multiplication:" << std::endl;
            timer t;
            auto C = new float[size * size];
            base_mul(C, A, B, size, size, size);
            t.end_timer();
            // checksum to prevent optimizer removing computation
            double trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            p_base = C;
            // delete[] C;
        }

        {
            std::cout << "Block multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_block(A, B, size, size, size);
            t.end_timer();
            double trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            std::cout << "Comparison: " << comp(C, p_base, size) << std::endl;
            delete[] C;
        }

        {
            std::cout << "Transpose multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_trans(A, B, size, size, size);
            t.end_timer();
            double trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            std::cout << "Comparison: " << comp(C, p_base, size) << std::endl;
            delete[] C;
        }

        {
            std::cout << "Transpose + Block multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_trans_block(A, B, size, size, size);
            t.end_timer();
            double trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            std::cout << "Comparison: " << comp(C, p_base, size) << std::endl;
            delete[] C;
        }

        {
            std::cout << "NEON multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_trans_block_with_neon(A, B, size, size, size);
            t.end_timer();
            double trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            std::cout << "Comparison: " << comp(C, p_base, size) << std::endl;
            free(C);
        }

        {
            std::cout << "Parallel Block multiplication (thread pool):" << std::endl;
            timer t;
            auto C = matrix_mul_block_parallel(A, B, size, size, size, 32, 0, 64);
            t.end_timer();
            double trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            std::cout << "Comparison: " << comp(C, p_base, size) << std::endl;
            delete[] C;
        }
    free(A);
    free(B);
    delete[] p_base;
    }
    return 0;
}