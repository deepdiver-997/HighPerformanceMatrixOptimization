#pragma once

#include <condition_variable>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <future>

class ThreadPool {
    std::vector<std::thread> workers;               // 工作线程
    std::queue<std::function<void()>> global_queue; // 全局任务队列
    std::mutex global_queue_mutex; // 全局任务队列互斥锁
    std::condition_variable condition;              // 任务到达通知
    std::atomic<bool> stop;                         // 停止标志

    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    public:
    static ThreadPool& get_instance(size_t num_threads = std::thread::hardware_concurrency()) {
        static ThreadPool instance(num_threads);
        return instance;
    }
    ~ThreadPool();
    // 提交任务到线程池
    void enqueue(std::function<void()> task);
    template<typename Func, typename... Args>
    auto enqueue_task(Func&& f, Args&&... args)
        -> std::future<typename std::invoke_result<Func, Args...>::type> {
        using return_type = typename std::invoke_result<Func, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(global_queue_mutex);
            global_queue.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }
    // 等待所有任务完成
    void wait_for_completion();
};

ThreadPool::ThreadPool(size_t num_threads)
    : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->global_queue_mutex);
                    this->condition.wait(lock, [this]() {
                        return this->stop.load() || !this->global_queue.empty();
                    });
                    if (this->stop.load() && this->global_queue.empty())
                        return;
                    task = std::move(this->global_queue.front());
                    this->global_queue.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (std::thread &worker : workers) {
        if (worker.joinable())
            worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(global_queue_mutex);
        global_queue.emplace(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::wait_for_completion() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(global_queue_mutex);
            if (global_queue.empty()) {
                break;
            }
        }
        std::this_thread::yield();
    }
}