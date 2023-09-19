#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class ThreadPool {
public:
  static constexpr std::size_t kDefaultStackSize{8 * 1024 * 1024};

public:
  static void* entry_point(void* context);

public:
  explicit ThreadPool(std::size_t threads);

  template <class F>
  ThreadPool(std::size_t threads, F &&initialize,
             std::size_t stack_size = kDefaultStackSize);

  template <class F, class... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  std::size_t thread_count() const;
  ~ThreadPool();

private:
  // perform initialization
  template <class F>
  void setup(std::size_t num_threads, F &&initialize, std::size_t stack_size);

  // need to keep track of threads so we can join them
  std::vector<pthread_t> threads;
  std::vector<std::function<void()>> entry_points;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

inline void* ThreadPool::entry_point(void *context) {
  auto * const callback{static_cast<std::function<void()>*>(context)};
  (*callback)();
  return nullptr;
}

template <class F>
inline void ThreadPool::setup(std::size_t num_threads, F &&initialize,
                              std::size_t stack_size) {
  threads.reserve(num_threads);
  entry_points.reserve(num_threads);

  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back();
    entry_points.emplace_back([initialize, this] {
      initialize();
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty()) {
            return;
          }
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        task();
      }
    });

    pthread_attr_t pthread_attribute;
    pthread_attr_init(&pthread_attribute);
    pthread_attr_setstacksize(&pthread_attribute, stack_size);

    pthread_create(&threads.back(), &pthread_attribute,
                   &ThreadPool::entry_point, &entry_points.back());

    pthread_attr_destroy(&pthread_attribute);
  }
}

// the constructor just launches some amount of workers and calls the
// initializer in each
template <class F>
inline ThreadPool::ThreadPool(std::size_t threads, F &&initialize,
                              std::size_t stack_size)
    : stop(false) {
  setup(threads, std::forward<F>(initialize), stack_size);
}

// this constructor just launches the workers.
inline ThreadPool::ThreadPool(std::size_t threads)
    : ThreadPool(threads, []() {}) {}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    const std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    const std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (const pthread_t &thread : threads) {
    pthread_join(thread, nullptr);
  }
}

inline std::size_t ThreadPool::thread_count() const { return threads.size(); }

#endif
