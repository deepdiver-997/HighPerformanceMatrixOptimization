#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <queue>
#include <functional>
#include <condition_variable>

class ThreadPool {
    std::vector<std::thread> workers;               // 工作线程
    std::vector<std::queue<std::function<void()>>> task_queues; // 每个线程的任务队列
    // store mutexes behind indirection because std::mutex is non-movable/non-copyable
    std::vector<std::unique_ptr<std::mutex>> queue_mutex;                         // 任务队列互斥锁
    std::queue<std::function<void()>> global_queue; // 全局任务队列
    std::mutex global_queue_mutex; // 全局任务队列互斥锁
    std::condition_variable condition;              // 任务到达通知
    std::atomic<bool> stop;                         // 停止标志
    std::atomic<int> next_queue;                             // 下一个任务队列索引
    int max_queue_size = 1024; // 每个队列的最大任务数
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency(), int mqs = 1024);
    ~ThreadPool();

    // 提交任务到线程池
    void enqueue(std::function<void()> task);

    bool thief_task(int thief_id) {
        {
            std::unique_lock<std::mutex> lock(global_queue_mutex);
            if (!global_queue.empty()) {
                std::unique_lock<std::mutex> llock(*queue_mutex[thief_id]);
                for (size_t i = 0; i < global_queue.size() / workers.size(); ++i) {
                    task_queues[thief_id].emplace(std::move(global_queue.front()));
                    global_queue.pop();
                }
                return true;
            }
        }
        for (size_t i = 0; i < task_queues.size(); ++i) {
            if (i == static_cast<size_t>(thief_id)) continue;
            std::unique_lock<std::mutex> lock(*queue_mutex[i]);
            if (task_queues[i].size() > max_queue_size / 2) {
                std::unique_lock<std::mutex> llock(*queue_mutex[thief_id]);
                for (size_t j = 0; j < task_queues[i].size() / workers.size(); ++j) {
                    task_queues[thief_id].emplace(std::move(task_queues[i].front()));
                    task_queues[i].pop();
                }
                return true;
            }
        }
        return false;
    }

    template<typename Func, typename... Args>
    auto enqueue_task(Func&& f, Args&&... args)
        -> std::future<typename std::invoke_result<Func, Args...>::type> {
        using return_type = typename std::invoke_result<Func, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        int n = next_queue.fetch_add(1) % task_queues.size();
        {
            std::unique_lock<std::mutex> lock(*queue_mutex[n]);
            if (task_queues[n].size() >= static_cast<size_t>(max_queue_size)) {
                // 如果队列已满，放入全局队列
                std::unique_lock<std::mutex> glock(global_queue_mutex);
                global_queue.emplace([task]() { (*task)(); });
                condition.notify_one();
                return res;
            }
            task_queues[n].emplace([task]() { (*task)(); });
            ++next_queue;
        }
        condition.notify_one();
        return res;
    }

    // 等待所有任务完成
    void wait_for_completion();
};

ThreadPool::ThreadPool(size_t num_threads, int mqs)
    : stop(false), next_queue(0), max_queue_size(mqs) {
    task_queues.resize(num_threads);
    queue_mutex.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        queue_mutex.emplace_back(std::make_unique<std::mutex>());
        workers.emplace_back([this, i]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(*this->queue_mutex[i]);
                    this->condition.wait(lock, [this, i]() {
                        return this->stop.load() || this->thief_task(i);
                    });
                    if (this->stop.load() && this->task_queues[i].empty())
                        return;
                    task = std::move(this->task_queues[i].front());
                    this->task_queues[i].pop();
                }
                task();
            }
        });
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    int n = next_queue.fetch_add(1) % task_queues.size();
    {
        std::unique_lock<std::mutex> lock(*queue_mutex[n]);
        if (task_queues[n].size() >= static_cast<size_t>(max_queue_size)) {
            // 如果队列已满，放入全局队列
            std::unique_lock<std::mutex> glock(global_queue_mutex);
            global_queue.emplace(std::move(task));
            condition.notify_one();
            return;
        }
        task_queues[n].emplace(std::move(task));
        ++next_queue;
    }
    condition.notify_one();
}

void ThreadPool::wait_for_completion() {
    for (size_t i = 0; i < workers.size(); ++i) {
        {
            std::unique_lock<std::mutex> lock(*this->queue_mutex[i]);
            this->condition.wait(lock, [this, i]() {
                return this->task_queues[i].empty();
            });
        }
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