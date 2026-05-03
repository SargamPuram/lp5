# Parallel Reduction — Min, Max, Sum, Average using OpenMP
### LP5 | Group A | Assignment 3

---

> **How to use:** Read question, cover answer, say it out loud.
> If you can say it confidently without reading — you're ready.

---

## SECTION 1 — Core Concept: What is Reduction?

---

### Q. What is Reduction in parallel computing?

Reduction is the process of combining all elements of an array into a single result using an associative operator — like finding the sum, minimum, maximum, or product. In sequential code this is a simple loop. In parallel computing, each thread works on a subset of the data independently, computes a local result, and then all local results are combined into the final answer.

---

### Q. Why is reduction needed in parallel programming?

If multiple threads all try to update one shared variable simultaneously — like all adding to the same `sum` variable — it causes a race condition. The final result would be wrong because thread updates overwrite each other. Reduction solves this by giving each thread its own private copy of the variable, letting them work independently, and then safely combining results at the end.

---

### Q. What is the reduction clause in OpenMP?

```cpp
#pragma omp parallel for reduction(operator : variable)
```

This directive tells OpenMP to:
1. Give each thread its own **private copy** of the variable, initialized to the identity value for that operator
2. Let each thread update its private copy independently — no locks needed
3. After all threads finish, **combine** all private copies using the operator into the final shared variable

---

### Q. What are the identity values for each operator?

| Operator | Identity Value | Why |
|---|---|---|
| `+` (sum) | 0 | x + 0 = x |
| `min` | INT_MAX | min(x, INT_MAX) = x |
| `max` | INT_MIN | max(x, INT_MIN) = x |
| `*` (product) | 1 | x × 1 = x |
| `&&` | 1 (true) | x && true = x |
| `\|\|` | 0 (false) | x \|\| false = x |

Each thread's private copy starts at this identity value so partial results are correct before merging.

---

### Q. What does associative mean and why does it matter for reduction?

An operation is associative if the order of grouping doesn't affect the result:
`(a + b) + c = a + (b + c)`

Min, max, sum, product are all associative. This means it doesn't matter which thread processes which elements — the final combined result will always be correct. Non-associative operations (like floating point subtraction) cannot be safely parallelized with reduction.

---

## SECTION 2 — How Each Operation Works

---

### Q. How does parallel minimum work?

```cpp
int minValue = arr[0];

#pragma omp parallel for reduction(min : minValue)
for (size_t i = 0; i < arr.size(); i++) {
    if (arr[i] < minValue)
        minValue = arr[i];
}
```

Each thread gets a private `minValue` initialized to `INT_MAX`. Each thread scans its assigned portion and updates its local minimum. After all threads finish, OpenMP takes the minimum of all thread-local minimums — giving the global minimum.

---

### Q. How does parallel maximum work?

```cpp
int maxValue = arr[0];

#pragma omp parallel for reduction(max : maxValue)
for (size_t i = 0; i < arr.size(); i++) {
    if (arr[i] > maxValue)
        maxValue = arr[i];
}
```

Same pattern as min but reversed. Each thread's private copy starts at `INT_MIN`. Each thread finds its local max. OpenMP combines by taking the maximum across all threads.

---

### Q. How does parallel sum work?

```cpp
long long sum = 0;

#pragma omp parallel for reduction(+ : sum)
for (size_t i = 0; i < arr.size(); i++) {
    sum += arr[i];
}
```

Each thread gets private `sum = 0`. Each thread adds its assigned elements to its local sum. OpenMP combines by adding all thread sums together. No critical section needed — completely race-condition free.

---

### Q. How does parallel average work?

Average cannot be directly reduced because there is no `average` operator. We compute it in two steps:

```cpp
long long sum = 0;

#pragma omp parallel for reduction(+ : sum)
for (size_t i = 0; i < arr.size(); i++) {
    sum += arr[i];
}

double avg = (double)sum / arr.size();  // Sequential step after parallel sum
```

The sum is computed in parallel. The division by n is done sequentially after — it's a single O(1) operation so no parallelism needed there.

---

### Q. Why do we use long long for sum instead of int?

For large arrays with large values, the sum can exceed the range of int (about 2.1 billion). For example, 1 million elements each up to 1000 gives a sum up to 1 billion — close to int limit. Using `long long` (range ~9.2 × 10^18) prevents integer overflow and incorrect results.

---

## SECTION 3 — OpenMP Reduction Internals

---

### Q. What exactly happens under the hood when reduction is used?

OpenMP internally does roughly this:

```
// What you write:
#pragma omp parallel for reduction(+ : sum)
for (int i = 0; i < n; i++) sum += arr[i];

// What OpenMP generates (conceptually):
#pragma omp parallel
{
    long long private_sum = 0;        // each thread gets this
    
    #pragma omp for
    for (int i = 0; i < n; i++)
        private_sum += arr[i];        // thread works on its own copy
    
    #pragma omp critical
    sum += private_sum;               // safely merge at the end
}
```

The critical section at the merge step is handled automatically by OpenMP — you don't write it yourself.

---

### Q. What is the difference between using reduction vs critical section for sum?

**With critical section (wrong approach):**
```cpp
long long sum = 0;
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    #pragma omp critical
    sum += arr[i];      // EVERY iteration acquires lock — fully serialized
}
```
This is technically correct but completely defeats parallelism — every thread waits for every other thread at every single element. Slower than sequential.

**With reduction (correct approach):**
```cpp
#pragma omp parallel for reduction(+ : sum)
for (int i = 0; i < n; i++) sum += arr[i];
```
Threads work completely independently. Lock is acquired only once per thread at the very end during merge. Genuinely parallel.

---

### Q. Can we do min and max without the reduction clause?

Yes, but it requires a critical section which is much slower:

```cpp
// Without reduction — needs critical, much slower
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    #pragma omp critical
    {
        if (arr[i] < minValue) minValue = arr[i];
    }
}
```

The reduction clause is cleaner, faster, and less code. It was specifically added to OpenMP to handle exactly this pattern.

---

### Q. Why is reduction faster than a critical section approach?

| | Critical Section | Reduction |
|---|---|---|
| Lock acquired | Once per element (n times) | Once per thread (p times) |
| Threads waiting | Constantly | Only at final merge |
| Parallel work | Almost none | Full array in parallel |
| Speedup | Often slower than sequential | Close to linear speedup |

For n=1,000,000 and p=4 threads: critical = 1,000,000 lock acquisitions vs reduction = 4 lock acquisitions.

---

## SECTION 4 — Performance

---

### Q. What performance did you observe in your benchmarks?

On 1,000,000 elements:
- Sequential: ~7888 microseconds
- Parallel Min: ~1616 microseconds
- Parallel Max: ~1749 microseconds  
- Parallel Sum: ~2165 microseconds
- Parallel Avg: ~2220 microseconds

Parallel was approximately **4x faster** than sequential — consistent with the number of CPU cores being used.

---

### Q. What are the performance characteristics of parallel reduction?

- **Small arrays** — parallel is slower due to thread creation overhead. Sequential wins.
- **Large arrays** — parallel gives near-linear speedup. 4 cores → ~4x faster.
- **Speedup** = Sequential Time / Parallel Time. Should approach number of cores.
- **Efficiency** = Speedup / Threads. Drops slightly due to memory bandwidth limits.
- Memory bandwidth is often the real bottleneck — all threads reading from same RAM.

---

### Q. What are the benefits of parallel reduction on large arrays?

- Near-linear speedup with number of cores
- No race conditions — each thread has private copy
- No critical sections in the hot loop — zero lock contention
- Simple to implement — just one clause on existing loop
- Works for many operators — sum, min, max, product, logical AND/OR

---

### Q. What are the limitations of parallel reduction?

- Thread creation overhead makes it slower on small arrays
- Memory bandwidth becomes bottleneck on large arrays — all threads compete for RAM access
- Only works for associative operators — cannot use for operations where order matters
- Floating point results may differ slightly due to different thread groupings (floating point is not strictly associative)
- Speedup limited by Amdahl's Law — sequential portions like the final division in average limit total gain

---

### Q. How does performance vary with input size?

| Array Size | Sequential | Parallel | Speedup |
|---|---|---|---|
| 100 | ~1 µs | ~10 µs | <1x (overhead dominates) |
| 10,000 | ~50 µs | ~30 µs | ~1.5x |
| 1,000,000 | ~7888 µs | ~1800 µs | ~4x |
| 100,000,000 | ~800ms | ~200ms | ~4x |

Speedup stabilizes near the number of CPU cores for large arrays.

---

## SECTION 5 — Official Viva Questions

---

### Q. What are the benefits of using parallel reduction for basic operations on large arrays?

Parallel reduction divides the array among multiple threads. Each thread computes a partial result on its assigned portion independently with no synchronization. Only at the end are results merged — with just one lock acquisition per thread. This gives near-linear speedup for large arrays. For 1 million elements we observed approximately 4x speedup. Additional benefits are simplicity — the reduction clause handles thread safety automatically — and correctness — no race conditions possible because each thread works on private data.

---

### Q. How does OpenMP's reduction clause work in parallel reduction?

The reduction clause gives each thread a private copy of the specified variable initialized to the identity value for the operator — 0 for sum, INT_MAX for min, INT_MIN for max. Each thread updates its private copy independently across its assigned iterations with no locking. After all threads finish the loop, OpenMP performs a tree-based or sequential merge combining all private copies using the specified operator to produce the final value in the original shared variable.

---

### Q. How do you set up a C++ program for parallel computation with OpenMP?

Three steps. First include the OpenMP header `#include <omp.h>`. Second add the pragma directive before the loop you want to parallelize — `#pragma omp parallel for reduction(op:var)`. Third compile with the `-fopenmp` flag: `g++ -fopenmp program.cpp -o program`. OpenMP then automatically manages thread creation, work distribution, and result merging.

---

### Q. What are the performance characteristics of parallel reduction and how do they vary with input size?

For small arrays under a few thousand elements, parallel reduction is slower than sequential because thread creation and synchronization overhead exceeds the actual computation savings. As input size grows beyond tens of thousands of elements, parallel reduction begins to show speedup. For millions of elements the speedup approaches the number of CPU cores — 4 cores give roughly 4x speedup. Beyond a certain size, memory bandwidth becomes the bottleneck and speedup plateaus regardless of thread count.

---

### Q. How can you modify the code for more complex operations using parallel reduction?

Several ways. First you can combine multiple reductions in one pragma — `reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)` — computing all three in a single pass over the array for better cache efficiency. Second you can use custom reduction for operations not built into OpenMP using `#pragma omp declare reduction`. Third you can extend to standard deviation by first computing sum in parallel, then computing sum of squared differences in a second parallel pass. Fourth you can apply reduction to 2D arrays by flattening them or using nested parallel regions.

---

## SECTION 6 — Tricky Examiner Questions

---

### Q. Why did you initialize minValue to arr[0] and not INT_MAX?

Both approaches are correct. Initializing to arr[0] is safer for edge cases — if the array contains only INT_MAX values, starting at arr[0] gives the correct answer. Internally OpenMP initializes each thread's private copy to INT_MAX regardless of what we set the shared variable to. The reduction merges private copies with the shared variable at the end, so starting at arr[0] gives a correct upper bound.

---

### Q. Can you compute min and max in a single parallel pass?

Yes — OpenMP allows multiple reduction clauses on one pragma:

```cpp
int minVal = arr[0], maxVal = arr[0];

#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
for (int i = 0; i < n; i++) {
    if (arr[i] < minVal) minVal = arr[i];
    if (arr[i] > maxVal) maxVal = arr[i];
}
```

This is more cache-efficient — the array is read once instead of twice.

---

### Q. Why can't you just use atomic instead of reduction for sum?

Atomic would work but be slower:

```cpp
// Atomic — correct but slow
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    #pragma omp atomic
    sum += arr[i];
}
```

Every single iteration requires an atomic hardware instruction — essentially serialized. Reduction accumulates into a private variable (zero overhead) and uses atomic or critical only once at the merge step per thread. For n=1,000,000, atomic = 1,000,000 atomic operations vs reduction = 4 atomic operations.

---

### Q. What is the difference between parallel reduction and MapReduce?

| | OpenMP Reduction | MapReduce |
|---|---|---|
| Architecture | Shared memory — one machine | Distributed — many machines |
| Communication | Through RAM | Through network |
| Scale | Cores on one CPU | Thousands of machines |
| Use case | HPC, scientific computing | Big data — Hadoop, Spark |
| Overhead | Microseconds | Seconds to minutes |

Both use the same fundamental idea — divide work, compute locally, combine results. OpenMP does it on one machine's cores. MapReduce does it across a cluster.

---

### Q. Is floating point reduction guaranteed to give the same result as sequential?

No. Floating point addition is not strictly associative due to rounding errors at each step. Different thread groupings produce slightly different intermediate rounding, giving slightly different final results. For example sequential sum may give 499880030.000001 while parallel gives 499880029.999998. Both are correct within floating point precision. For exact reproducibility, sequential is needed. This is a known limitation of parallel floating point reduction.

---

### Q. What happens if the array has only one element?

All operations still work correctly. The parallel for loop runs once with one iteration assigned to one thread. Min, max, sum, average all return that single element's value. The code handles this safely because we initialize to arr[0] and the loop processes from index 0.

---

### Q. Can reduction be used with while loops or only for loops?

Only with for loops in OpenMP. The omp parallel for directive requires a loop with a known iteration count that can be divided among threads. While loops have dynamic termination conditions — OpenMP cannot statically distribute iterations. For while-loop-based reductions you would need to manually use omp parallel with private variables and a critical section at the merge.

---

## SECTION 7 — Quick Revision Flashcards

| Question | One line answer |
|---|---|
| What is reduction? | Combine all elements into one result using an associative operator |
| Why reduction instead of shared variable? | Shared variable causes race condition — all threads overwriting same memory |
| What does reduction clause do? | Gives each thread private copy, merges at end automatically |
| Identity value for sum? | 0 |
| Identity value for min? | INT_MAX |
| Identity value for max? | INT_MIN |
| Why long long for sum? | Prevent overflow for large arrays with large values |
| How is average computed in parallel? | Sum in parallel, divide by n sequentially after |
| Why reduction faster than critical? | Critical locks n times, reduction locks p times (p = threads) |
| Why reduction faster than atomic? | Atomic per element vs atomic only at final merge |
| When is parallel slower than sequential? | Small arrays — thread overhead dominates actual work |
| What is speedup on 1M elements? | ~4x — matches number of CPU cores |
| What limits speedup on huge arrays? | Memory bandwidth — all threads competing for RAM |
| What is Amdahl's Law? | Max speedup limited by sequential portion of code |
| Can you reduce floating point exactly? | No — rounding differs with different thread groupings |
| Works with while loops? | No — only for loops with known iteration count |
| Compile command? | g++ -fopenmp program.cpp -o program |
| Can multiple reductions on one pragma? | Yes — reduction(min:a) reduction(max:b) on same line |

---

*Reduction is the cleanest OpenMP topic — one clause does everything.*
*Key message: private copies + merge at end = no race condition + real parallelism.*