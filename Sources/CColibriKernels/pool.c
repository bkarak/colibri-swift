// A spinning worker pool for the CPU backend's parallel loops.
//
// A decode step is a few hundred short parallel loops (0.1-0.5 ms each). With
// `DispatchQueue.concurrentPerform` the workers park between loops and each loop pays for
// waking them: on Qwen3-4B that alone held a token's matmuls to about 20 tok/s against
// about 50 for the same kernels with no wake-ups. Here the workers spin on a generation
// counter between loops, then sleep after `SPIN_NS` without work, so an idle process
// does not keep the cores busy.
//
// Work is handed out dynamically: one 64-bit atomic holds (generation << 32 | next chunk),
// and a worker claims a chunk with a compare-and-swap that also checks the generation, so
// it never runs a chunk of a loop that has already finished. The caller takes chunks too
// and returns when every chunk has completed. One loop at a time: a caller that finds the
// pool busy (another thread's loop, or a worker calling in) runs its loop serially.
#include "colibri_kernels.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <time.h>
#include <mach/mach_time.h>

#define MAX_WORKERS 63
#define SPIN_NS 3000000ull  // 3 ms of spinning before a worker sleeps

static struct {
    pthread_mutex_t submit;            // held by the thread running a loop
    pthread_mutex_t sleep_lock;
    pthread_cond_t wake;
    _Atomic uint64_t claim;            // generation << 32 | next chunk
    _Atomic uint32_t generation;       // bumped when a loop is posted
    _Atomic uint32_t completed;        // chunks finished in the current loop
    _Atomic int sleepers;
    colibri_chunk_fn fn;               // stable while a loop runs
    void *ctx;
    _Atomic uint32_t chunks;
    int workers;                       // started workers
} pool = {
    .submit = PTHREAD_MUTEX_INITIALIZER,
    .sleep_lock = PTHREAD_MUTEX_INITIALIZER,
    .wake = PTHREAD_COND_INITIALIZER,
};

static _Thread_local int is_worker;
static mach_timebase_info_data_t timebase;

static uint64_t now_ns(void) { return mach_absolute_time() * timebase.numer / timebase.denom; }

/// Claims and runs chunks of loop `gen` until none are left; returns how many it ran.
static uint32_t drain(uint32_t gen) {
    uint32_t ran = 0;
    for (;;) {
        uint64_t c = atomic_load_explicit(&pool.claim, memory_order_acquire);
        if ((uint32_t)(c >> 32) != gen || (uint32_t)c >= atomic_load_explicit(&pool.chunks, memory_order_relaxed)) break;
        if (!atomic_compare_exchange_weak_explicit(&pool.claim, &c, c + 1, memory_order_acq_rel, memory_order_acquire))
            continue;
        // The claim succeeded for `gen`, so that loop is still running and its fields
        // are stable until this chunk completes.
        pool.fn(pool.ctx, (int)(uint32_t)c);
        atomic_fetch_add_explicit(&pool.completed, 1, memory_order_release);
        ran++;
    }
    return ran;
}

static void *worker_main(void *arg) {
    (void)arg;
    is_worker = 1;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint32_t seen = atomic_load_explicit(&pool.generation, memory_order_acquire);
    for (;;) {
        uint64_t idle_since = now_ns();
        uint32_t g;
        int spins = 0;
        while ((g = atomic_load_explicit(&pool.generation, memory_order_acquire)) == seen) {
            __builtin_arm_yield();
            if (++spins < 4096) continue;
            spins = 0;
            if (now_ns() - idle_since < SPIN_NS) continue;
            pthread_mutex_lock(&pool.sleep_lock);
            atomic_fetch_add(&pool.sleepers, 1);
            while (atomic_load_explicit(&pool.generation, memory_order_acquire) == seen)
                pthread_cond_wait(&pool.wake, &pool.sleep_lock);
            atomic_fetch_sub(&pool.sleepers, 1);
            pthread_mutex_unlock(&pool.sleep_lock);
            idle_since = now_ns();
        }
        seen = g;
        drain(g);
    }
    return NULL;
}

static void start_workers(int want) {
    if (want > MAX_WORKERS) want = MAX_WORKERS;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    while (pool.workers < want) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
        pthread_t t;
        if (pthread_create(&t, &attr, worker_main, NULL) != 0) {
            pthread_attr_destroy(&attr);
            break;
        }
        pthread_detach(t);
        pthread_attr_destroy(&attr);
        pool.workers++;
    }
}

void colibri_parallel_for(int chunks, int threads, colibri_chunk_fn fn, void *ctx) {
    if (chunks <= 0) return;
    if (chunks == 1 || threads <= 1 || is_worker || pthread_mutex_trylock(&pool.submit) != 0) {
        for (int i = 0; i < chunks; i++) fn(ctx, i);
        return;
    }
    start_workers(threads - 1);
    pool.fn = fn;
    pool.ctx = ctx;
    atomic_store_explicit(&pool.chunks, (uint32_t)chunks, memory_order_relaxed);
    atomic_store_explicit(&pool.completed, 0, memory_order_relaxed);
    uint32_t gen = atomic_load_explicit(&pool.generation, memory_order_relaxed) + 1;
    atomic_store_explicit(&pool.claim, (uint64_t)gen << 32, memory_order_release);
    atomic_store_explicit(&pool.generation, gen, memory_order_release);
    if (atomic_load(&pool.sleepers) > 0) {
        pthread_mutex_lock(&pool.sleep_lock);
        pthread_cond_broadcast(&pool.wake);
        pthread_mutex_unlock(&pool.sleep_lock);
    }
    drain(gen);
    while (atomic_load_explicit(&pool.completed, memory_order_acquire) < (uint32_t)chunks) __builtin_arm_yield();
    pthread_mutex_unlock(&pool.submit);
}
