#include "source/parallel/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace seeml::update {

namespace {

// Pool workers execute nested ParallelFor calls inline: the outer call
// already owns the pool, and a worker blocking on its own pool would
// deadlock.
thread_local bool tls_pool_worker = false;

// Same rule for the submitting thread itself: while it owns the in-flight
// job (and the submission lock), a nested ParallelFor from one of its own
// chunk bodies must execute inline — re-submitting would self-deadlock on
// the non-recursive submission lock.
thread_local bool tls_job_owner = false;

size_t ResolveAutoThreadCount() {
  if (const char* env = std::getenv("SEEML_THREADS")) {
    // strtoll, not strtoul: strtoul silently accepts "-1" and wraps it to
    // ULONG_MAX, which would become the pool width. Negative, overflowing,
    // or malformed values all fall back to hardware_concurrency.
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(env, &end, 10);
    if (end != env && *end == '\0' && errno != ERANGE && v > 0)
      return static_cast<size_t>(v);
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? hw : 1;
}

// One in-flight parallel loop. Lives on the calling thread's stack for the
// duration of Run(): `holders` counts workers that still reference it, and
// the caller does not return until every holder has detached — no worker
// ever touches a dead Job.
struct Job {
  internal_par::ChunkFn fn = nullptr;
  void* ctx = nullptr;
  size_t n = 0;
  size_t grain = 0;
  size_t chunks = 0;
  // On separate cache lines: `next` is hammered by every chunk claim.
  alignas(64) std::atomic<size_t> next{0};
  alignas(64) std::atomic<size_t> done{0};
  alignas(64) std::atomic<size_t> holders{0};
  // First exception thrown by a chunk body. Captured here so DrainJob never
  // throws: an unwind out of a worker would call std::terminate, and an
  // unwind out of Run() would destroy this stack-allocated Job while workers
  // still hold pointers to it. Rethrown by Run() after the loop retires.
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::exception_ptr error;
};

void DrainJob(Job& job) {
  for (;;) {
    const size_t c = job.next.fetch_add(1, std::memory_order_relaxed);
    if (c >= job.chunks) return;
    // Once a body has thrown, remaining chunks are claimed but not executed:
    // `done` still reaches `chunks`, so the submitter's wait always finishes.
    if (!job.failed.load(std::memory_order_relaxed)) {
      const size_t begin = c * job.grain;
      const size_t end = std::min(begin + job.grain, job.n);
      try {
        job.fn(job.ctx, begin, end, c);
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(job.error_mutex);
          if (!job.error) job.error = std::current_exception();
        }
        job.failed.store(true, std::memory_order_relaxed);
      }
    }
    // Release pairs with the caller's acquire load of `done`: chunk results
    // are visible before the loop is observed complete.
    job.done.fetch_add(1, std::memory_order_acq_rel);
  }
}

// Persistent worker pool. Intentionally leaked: workers park on the
// condition variable across calls and the process teardown reclaims them;
// a static destructor racing late ParallelFor calls is a worse failure mode
// than an intentionally immortal singleton.
class WorkerPool {
 public:
  static WorkerPool& Instance() {
    static WorkerPool* pool = new WorkerPool;
    return *pool;
  }

  size_t ThreadCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_count_ == 0) thread_count_ = ResolveAutoThreadCount();
    return thread_count_;
  }

  void SetThreadCount(size_t count) {
    JoinWorkers();
    std::lock_guard<std::mutex> lock(mutex_);
    thread_count_ = count > 0 ? count : ResolveAutoThreadCount();
  }

  void Run(Job& job) {
    // One job in flight at a time: concurrent external submitters queue here
    // in arrival order, each getting the whole pool. Without this, a second
    // submitter would overwrite job_/epoch_ (stealing the first job's
    // workers) and the first submitter's cleanup would null out the second's
    // freshly published job.
    std::lock_guard<std::mutex> submit(submit_mutex_);
    // RAII: an exception escaping anything below (worker spawn is the
    // fallible step) must not leave the flag set, or this thread would run
    // the inline path forever after one transient failure.
    struct OwnerFlag {
      OwnerFlag() { tls_job_owner = true; }
      ~OwnerFlag() { tls_job_owner = false; }
    } owner_flag;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      SpawnWorkersLocked();
      job_ = &job;
      ++epoch_;
    }
    work_cv_.notify_all();
    // DrainJob never throws (body exceptions are captured in the Job), so
    // the holder wait below always runs and the Job outlives every worker
    // reference even when a chunk body failed.
    DrainJob(job);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&] {
        return job.done.load(std::memory_order_acquire) == job.chunks &&
               job.holders.load(std::memory_order_acquire) == 0;
      });
      job_ = nullptr;
    }
    if (job.error) std::rethrow_exception(job.error);
  }

 private:
  void SpawnWorkersLocked() {
    if (thread_count_ == 0) thread_count_ = ResolveAutoThreadCount();
    // The caller participates, so the pool holds thread_count_ - 1 workers.
    // Top-up, not spawn-once, and degrade instead of aborting: a
    // std::thread constructor throwing (thread/resource exhaustion) runs
    // this job at whatever width exists — chunk geometry is shape-based,
    // so results are identical — and the next Run retries the missing
    // workers instead of freezing the narrowed pool forever.
    while (workers_.size() + 1 < thread_count_) {
      try {
        workers_.emplace_back([this] { WorkerLoop(); });
      } catch (const std::system_error&) {
        break;
      }
    }
  }

  void JoinWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (workers_.empty()) return;
      stop_ = true;
    }
    work_cv_.notify_all();
    for (std::thread& t : workers_) t.join();
    std::lock_guard<std::mutex> lock(mutex_);
    workers_.clear();
    stop_ = false;
  }

  void WorkerLoop() {
    tls_pool_worker = true;
    uint64_t seen_epoch = 0;
    for (;;) {
      Job* job = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock,
                      [&] { return stop_ || epoch_ != seen_epoch; });
        if (stop_) return;
        seen_epoch = epoch_;
        job = job_;
        // Registered under the same lock that publishes job_, so the caller
        // cannot observe holders == 0 and retire the Job in between.
        if (job) job->holders.fetch_add(1, std::memory_order_relaxed);
      }
      if (!job) continue;
      DrainJob(*job);
      job->holders.fetch_sub(1, std::memory_order_acq_rel);
      {
        std::lock_guard<std::mutex> lock(mutex_);
      }
      done_cv_.notify_all();
    }
  }

  std::mutex submit_mutex_;  // serializes whole Run() calls; held first
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  Job* job_ = nullptr;        // guarded by mutex_
  uint64_t epoch_ = 0;        // guarded by mutex_
  size_t thread_count_ = 0;   // guarded by mutex_; 0 = not yet resolved
  bool stop_ = false;         // guarded by mutex_
};

}  // namespace

size_t ParallelThreadCount() { return WorkerPool::Instance().ThreadCount(); }

void SetParallelThreadCount(size_t count) {
  WorkerPool::Instance().SetThreadCount(count);
}

namespace internal_par {

void ParallelForRaw(size_t n, size_t grain, ChunkFn fn, void* ctx) {
  if (n == 0) return;
  const size_t g = ParallelChunkGrain(n, grain);
  const size_t chunks = (n + g - 1) / g;
  WorkerPool& pool = WorkerPool::Instance();
  if (chunks == 1 || tls_pool_worker || tls_job_owner ||
      pool.ThreadCount() == 1) {
    for (size_t c = 0; c < chunks; ++c)
      fn(ctx, c * g, std::min((c + 1) * g, n), c);
    return;
  }
  Job job;
  job.fn = fn;
  job.ctx = ctx;
  job.n = n;
  job.grain = g;
  job.chunks = chunks;
  pool.Run(job);
}

}  // namespace internal_par

}  // namespace seeml::update
