# Parallel Bubble Sort & Merge Sort using OpenMP — Complete Viva Guide
### LP5 | Group A | Assignment 2A & 2B

---

> **How to use:** Read each question, cover the answer, say it out loud.
> If you can say it confidently without reading — you're ready.

---

## SECTION 1 — Bubble Sort Basics

---

### Q. What is Bubble Sort?

Bubble Sort is a simple comparison-based sorting algorithm that repeatedly steps through the array, compares adjacent elements, and swaps them if they are in the wrong order. This process is repeated until no swaps are needed, meaning the array is sorted. It is called bubble sort because larger elements gradually bubble up to the end of the array with each pass, like bubbles rising in water.

---

### Q. What is the time and space complexity of Bubble Sort?

| Case | Time Complexity |
|---|---|
| Best case (already sorted) | O(n) |
| Average case | O(n²) |
| Worst case (reverse sorted) | O(n²) |
| Space complexity | O(1) — in place |

---

### Q. When is Bubble Sort useful despite being O(n²)?

- Educational purposes — simplest algorithm to understand and explain
- Very small datasets where overhead of complex algorithms is not worth it
- Nearly sorted data — very few swaps needed, approaches O(n)
- When simplicity of implementation matters more than performance

---

### Q. What is the basic algorithm of Bubble Sort step by step?

1. Start from the first element
2. Compare adjacent pair arr[j] and arr[j+1]
3. If arr[j] > arr[j+1], swap them
4. Move to the next pair
5. After one full pass, the largest element is at the end
6. Repeat for remaining n-1 elements
7. Stop when no swaps occur in a full pass

---

## SECTION 2 — Parallel Bubble Sort

---

### Q. What is Parallel Bubble Sort? How does it work?

Parallel Bubble Sort uses the Odd-Even Transposition technique. Instead of comparing all pairs sequentially, we split comparisons into two independent phases that can run in parallel:

- **Even phase** — compare and swap pairs at positions (0,1), (2,3), (4,5)... simultaneously
- **Odd phase** — compare and swap pairs at positions (1,2), (3,4), (5,6)... simultaneously

Within each phase, all comparisons are on completely separate pairs with no shared elements, so multiple threads can handle them simultaneously without any race condition.

---

### Q. Why can even and odd phases be parallelized safely?

In the even phase, thread 1 works on indices (0,1), thread 2 on (2,3), thread 3 on (4,5) — no two threads touch the same element. Same for odd phase. Since no shared data is accessed simultaneously, no critical section is needed. This is what makes it safe to use omp parallel for directly without any synchronization.

---

### Q. Why do we need both even AND odd phases?

Each phase alone cannot fully sort the array because adjacent pairs don't overlap within a phase. The odd phase handles the pairs that the even phase skipped. Together, alternating between even and odd phases ensures every adjacent pair gets compared across iterations, achieving complete sorting.

---

### Q. How is parallel bubble sort implemented using OpenMP?

```cpp
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {

        // Even phase — pairs (0,1), (2,3), (4,5)...
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }

        // Odd phase — pairs (1,2), (3,4), (5,6)...
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}
```

Each omp parallel for has an implicit barrier at the end — all threads finish the even phase completely before the odd phase starts. This ensures correctness.

---

### Q. Why is there no critical section in parallel bubble sort?

Because within each phase, every thread operates on completely separate non-overlapping index pairs. Thread 1 touches arr[0] and arr[1], thread 2 touches arr[2] and arr[3] — they never share an element within the same phase. No shared data = no race condition = no critical section needed.

---

### Q. What is the implicit barrier in omp parallel for?

At the end of every omp parallel for block, all threads automatically wait for each other before the program continues. This is called an implicit barrier. In parallel bubble sort this is critical — all threads must finish the even phase before any thread starts the odd phase. Without this guarantee the two phases would interfere.

---

### Q. Difference between sequential and parallel bubble sort?

| | Sequential Bubble Sort | Parallel Bubble Sort |
|---|---|---|
| Approach | One pair at a time | Multiple pairs simultaneously |
| Technique | Standard adjacent comparison | Odd-even transposition |
| Threads | 1 | Multiple (omp parallel for) |
| Time complexity | O(n²) | O(n²/p) where p = threads |
| Synchronization | None needed | Implicit barrier between phases |
| Speedup on small arrays | Faster (no overhead) | Slower (thread creation cost) |
| Speedup on large arrays | Baseline | Faster with multiple cores |

---

### Q. What are the advantages of parallel bubble sort?

- Utilizes multiple CPU cores — better hardware usage
- Speedup proportional to number of threads on large arrays
- No critical sections needed — low synchronization overhead
- Simple to implement using just omp parallel for
- Maintains correctness through the odd-even transposition technique

---

## SECTION 3 — Merge Sort Basics

---

### Q. What is Merge Sort?

Merge Sort is a divide-and-conquer sorting algorithm. It recursively divides the array into two halves, sorts each half, and then merges the two sorted halves into a single sorted array. The key insight is that merging two already-sorted arrays is an O(n) operation. This gives merge sort its O(n log n) time complexity which is optimal for comparison-based sorting.

---

### Q. What is the time and space complexity of Merge Sort?

| Case | Time Complexity |
|---|---|
| Best case | O(n log n) |
| Average case | O(n log n) |
worst case | O(n log n) |
| Space complexity | O(n) — needs auxiliary array for merging |

Merge sort is always O(n log n) regardless of input — unlike quicksort which degrades to O(n²) in the worst case.

---

### Q. What are the steps of Merge Sort?

1. If array has 1 or 0 elements — it is already sorted, return
2. Find the middle index m = (l + r) / 2
3. Recursively sort the left half arr[l..m]
4. Recursively sort the right half arr[m+1..r]
5. Merge the two sorted halves back into arr[l..r]

The merge step compares elements from both halves one by one and places the smaller one into the output array until both halves are exhausted.

---

### Q. Why is Merge Sort preferred over Bubble Sort for large data?

| | Bubble Sort | Merge Sort |
|---|---|---|
| Time complexity | O(n²) | O(n log n) |
| For n = 10,000 | ~100,000,000 operations | ~130,000 operations |
| Stable sort | Yes | Yes |
| In-place | Yes | No (needs O(n) extra space) |
| Practical use | Educational only | Production systems, large data |

---

## SECTION 4 — Parallel Merge Sort

---

### Q. What is Parallel Merge Sort and how does it work?

Parallel Merge Sort exploits the fact that the two recursive halves are completely independent of each other — sorting the left half has no effect on sorting the right half. So both halves can be sorted simultaneously by different threads.

We use omp parallel sections to assign the left and right recursive calls to separate threads. After both finish, we merge them sequentially (merge itself is harder to parallelize correctly).

---

### Q. How is parallel merge sort implemented using OpenMP?

```cpp
void mergeSortParallel(vector<int>& arr, int l, int r, int depth) {
    if (l < r) {
        int m = l + (r - l) / 2;

        if (depth <= 0) {
            // Too deep — fall back to sequential to avoid
            // thread explosion creating thousands of threads
            mergeSortSeq(arr, l, m);
            mergeSortSeq(arr, m + 1, r);
        } else {
            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortParallel(arr, l, m, depth - 1);

                #pragma omp section
                mergeSortParallel(arr, m + 1, r, depth - 1);
            }
        }
        merge(arr, l, m, r);   // Always sequential — merge two sorted halves
    }
}
```

---

### Q. What is the depth parameter in parallel merge sort and why is it needed?

The depth parameter controls how many levels of recursion use parallelism. We call it with depth = 4, meaning only the top 4 levels spawn parallel sections. Below that, recursion falls back to sequential.

Without depth control, every recursive call would create a new parallel sections block. A 50,000 element array has about 16 levels of recursion — that would spawn 2^16 = 65,536 thread teams. Thread creation overhead would completely dominate and the code would be drastically slower than sequential. Depth = 4 creates at most 2^4 = 16 parallel sections — enough parallelism without excess overhead.

---

### Q. Why is the merge step sequential and not parallelized?

Merging two sorted arrays requires reading both halves from left to right in order — the output of each comparison depends on the previous one. There is no independent work to split across threads cleanly. Parallel merge algorithms exist but are significantly more complex, requiring additional memory and coordination. For this practical the merge is kept sequential since the main speedup comes from parallel recursive sorting anyway.

---

### Q. Why do we use sections for merge sort but tasks for DFS?

| | Merge Sort | DFS |
|---|---|---|
| Branches at each level | Always exactly 2 (left and right half) | Variable — depends on neighbor count |
| Known at compile time? | Yes — always exactly 2 sections | No — only known at runtime |
| Best OpenMP primitive | sections | task |

Merge sort always splits into exactly two halves — left and right. This is known at compile time, making sections a perfect fit. DFS branches depend on the number of unvisited neighbors which is only known during execution, making tasks necessary.

---

### Q. Difference between sequential and parallel merge sort?

| | Sequential Merge Sort | Parallel Merge Sort |
|---|---|---|
| Approach | One half at a time | Both halves simultaneously |
| OpenMP directive | None | omp parallel sections |
| Thread overhead | None | Yes — at upper recursion levels |
| Time complexity | O(n log n) | O(n log n / p) for top levels |
| Speedup on small arrays | Faster | Slower (overhead) |
| Speedup on large arrays | Baseline | Faster with multiple cores |

---

### Q. What are the advantages of parallel merge sort?

- Left and right halves are completely independent — natural parallelism
- Significant speedup on large arrays with multiple cores
- Always O(n log n) — no worst case degradation unlike quicksort
- Depth control prevents thread explosion
- Stable sort — relative order of equal elements preserved

---

## SECTION 5 — Performance Measurement

---

### Q. How do you measure performance of sequential vs parallel algorithms?

We use the chrono library in C++ for high precision timing:

```cpp
auto start = high_resolution_clock::now();
// run algorithm
auto end = high_resolution_clock::now();
cout << duration_cast<milliseconds>(end - start).count() << " ms";
```

Steps for fair comparison:
1. Use the same input array for all algorithms — copy it before each run
2. Use large enough input — small arrays don't show meaningful differences
3. Run multiple times and average — reduces noise from system load
4. Measure wall clock time — what the user actually experiences

---

### Q. What is speedup and how do you calculate it?

Speedup = Sequential Time / Parallel Time

- Speedup > 1 means parallel is faster
- Speedup = number of cores in ideal case (linear speedup)
- Speedup < 1 means parallel is slower — overhead dominates

Example: Sequential bubble sort = 528ms, Parallel = 420ms → Speedup = 528/420 = 1.26x

---

### Q. What is efficiency in parallel computing?

Efficiency = Speedup / Number of Threads

If speedup is 3x with 4 threads, efficiency = 3/4 = 75%. 
100% efficiency means perfect linear scaling — rarely achieved due to synchronization overhead, memory bandwidth limits, and sequential portions of the code (Amdahl's Law).

---

### Q. What is Amdahl's Law?

Amdahl's Law states that the maximum speedup of a parallel program is limited by its sequential portion. If 20% of the code cannot be parallelized, the maximum possible speedup is 1/0.2 = 5x regardless of how many threads you use. In merge sort the merge step is sequential — this limits total speedup even if sorting is perfectly parallel.

---

### Q. Why is parallel bubble sort slower than sequential on our benchmarks?

On WSL with small-to-medium arrays, parallel bubble sort is slower because:
- Thread creation overhead from omp parallel for called 2×n times (once per phase per iteration)
- For n=10,000 that is 20,000 thread pool fork-joins
- Each fork-join on WSL costs microseconds — adds up to more than the sorting savings
- The actual comparison work per thread is trivial — not worth the overhead

On a native Linux machine with large arrays (100,000+), parallel bubble sort would show genuine speedup.

---

### Q. How do you check CPU utilization during program execution in Ubuntu?

```bash
# Real-time view of CPU and memory usage
top

# Better version with colors and interactivity
htop

# Snapshot of all processes
ps aux

# Memory usage summary
free -h

# CPU + memory + disk stats
vmstat
```

Run the sort program in one terminal and monitor with htop in another to visually see cores being used.

---

## SECTION 6 — Official Viva Questions

---

### Q. What is parallel Bubble Sort?

Parallel Bubble Sort is a modified bubble sort that uses the odd-even transposition technique. Instead of comparing one pair at a time, it simultaneously compares all even-indexed pairs in one phase and all odd-indexed pairs in the next phase using omp parallel for. Within each phase all comparisons are on non-overlapping pairs so no synchronization is needed. The two phases alternate for n iterations until the array is fully sorted.

---

### Q. How does parallel Bubble Sort work?

It works in two alternating phases per iteration. In the even phase, threads simultaneously compare and swap pairs at positions (0,1), (2,3), (4,5) and so on. In the odd phase, threads simultaneously compare and swap pairs at (1,2), (3,4), (5,6) and so on. An implicit barrier between phases ensures even phase completes before odd phase starts. After n such iterations the array is guaranteed to be sorted.

---

### Q. How do you implement parallel Bubble Sort using OpenMP?

Using omp parallel for on the inner loop with j incrementing by 2 for even phase starting at 0, and odd phase starting at 1. No critical section is needed because each thread works on non-overlapping index pairs. The outer loop runs n times to guarantee complete sorting.

---

### Q. What are the advantages of parallel Bubble Sort?

- Uses all available CPU cores
- No critical sections needed — zero lock contention
- Simple implementation — just add omp parallel for
- Speedup proportional to threads on large arrays
- Correct by design — odd-even transposition is a proven parallel sorting network

---

### Q. What is parallel Merge Sort?

Parallel Merge Sort parallelizes the divide step of merge sort. Since the left and right halves are completely independent, both can be sorted simultaneously using omp parallel sections. A depth parameter controls how many levels use parallelism to prevent thread explosion. Below the depth threshold, it falls back to sequential merge sort. The merge step itself remains sequential.

---

### Q. How does parallel Merge Sort work?

At each recursive call, if depth > 0, omp parallel sections assigns the left half to one thread and the right half to another. Both sort their halves simultaneously using recursive calls with depth-1. When both finish (implicit barrier at end of sections), the two sorted halves are merged sequentially. This continues until depth reaches 0, below which sequential merge sort handles all remaining recursion.

---

### Q. How do you implement parallel Merge Sort using OpenMP?

Using omp parallel sections with two sections — one for left half recursion and one for right half recursion. depth parameter starts at 4, allowing 2^4 = 16 parallel sections across the top levels of recursion. The merge function is unchanged from sequential — it simply merges two sorted subarrays by comparing elements one by one.

---

### Q. What are the advantages of parallel Merge Sort?

- Left and right halves are naturally independent — easy to parallelize
- Always O(n log n) — no worst case like quicksort
- Stable sort — preserves relative order of equal elements
- Significant speedup on large arrays
- Depth control makes it practical — avoids thread explosion

---

### Q. Difference between serial and parallel merge sort?

Serial processes left then right half one after the other. Parallel processes both halves simultaneously using sections. Serial has no thread overhead — faster on small arrays. Parallel utilizes multiple cores — faster on large arrays. Both produce identical sorted output. Both have O(n log n) time complexity but parallel reduces wall clock time by up to p times where p is number of threads.

---

## SECTION 7 — Tricky Examiner Questions

---

### Q. Why is parallel bubble sort still O(n²) if it uses multiple threads?

The number of comparisons doesn't change — we still do n passes with n comparisons each. Parallelism reduces wall clock time by doing comparisons simultaneously, but the total work remains O(n²). Parallel complexity is O(n²/p) for p threads, but asymptotically it is still O(n²). True algorithmic complexity improvement would require a different algorithm like merge sort.

---

### Q. Why doesn't parallel merge sort parallelize the merge step?

The merge step reads both halves sequentially from left to right — each decision depends on the previous one. There is no independent work to split cleanly. Parallel merge algorithms exist but require O(log n) extra passes and complex coordination, making them impractical for this level. The main speedup comes from parallel recursive sorting which already provides significant benefit.

---

### Q. What happens if you set depth = 0 in parallel merge sort?

It degrades completely to sequential merge sort. Every call immediately hits the depth <= 0 branch and calls mergeSortSeq for both halves. No parallelism occurs at all.

---

### Q. What happens if you set depth very high, like depth = 20?

For an array of n elements, merge sort has log₂(n) levels. Setting depth higher than log₂(n) means every single recursive level tries to spawn parallel sections. For 50,000 elements that is about 16 levels — potentially 2^16 = 65,536 thread teams. Thread creation overhead completely dominates and the program becomes much slower than sequential. Depth = 4 is a practical sweet spot giving 16 parallel sections without excess overhead.

---

### Q. Why does sequential merge sort beat parallel bubble sort even though bubble sort is parallelized?

Because they have fundamentally different time complexities. Bubble sort is O(n²) — even with p threads it is O(n²/p). Merge sort is O(n log n) sequentially. For large n, O(n log n) < O(n²/p) for any reasonable p. For n=50,000 and p=8 threads: bubble sort ≈ 312,500,000 operations, merge sort ≈ 780,000 operations. Parallelism cannot fix a bad algorithm — algorithm choice matters more than parallelism.

---

### Q. Is our parallel bubble sort truly parallel or does the barrier make it sequential?

The barrier between even and odd phases does introduce sequential waiting — even phase must fully complete before odd phase begins. But within each phase all comparisons happen truly in parallel. So it is partially parallel — parallel within each phase, sequential between phases. This is correct and necessary — if odd and even phases overlapped, an element could be involved in two swaps simultaneously causing incorrect results.

---

### Q. Why do we copy the array before each sort?

```cpp
temp = arr;  // fresh copy for each algorithm
bubbleSortSeq(temp);
temp = arr;  // reset
bubbleSortParallel(temp);
```

All four sorting algorithms must sort the same original unsorted array to make the benchmark fair. If we passed the already-sorted array to the second algorithm it would finish in O(n) instead of O(n²) — completely skewing the comparison. Each algorithm gets an identical fresh unsorted copy.

---

## SECTION 8 — Quick Revision Flashcards

| Question | One line answer |
|---|---|
| What is bubble sort? | Repeatedly swap adjacent elements until sorted |
| Time complexity bubble sort? | O(n²) worst and average, O(n) best |
| What is merge sort? | Divide into halves, sort each, merge back |
| Time complexity merge sort? | O(n log n) always |
| How is bubble sort parallelized? | Odd-even transposition — even pairs then odd pairs |
| Why no critical section in parallel bubble? | Each thread works on non-overlapping index pairs |
| What is implicit barrier? | All threads wait at end of omp parallel for automatically |
| How is merge sort parallelized? | omp sections — left and right halves simultaneously |
| What is depth parameter? | Controls how many recursion levels use parallelism |
| Why depth limit in merge sort? | Prevents thread explosion — too many teams = overhead |
| What is speedup? | Sequential time / Parallel time |
| What is efficiency? | Speedup / Number of threads |
| What is Amdahl's Law? | Max speedup limited by sequential portion of code |
| Why copy array before each sort? | Fair comparison — all algorithms sort same unsorted input |
| Why parallel bubble slower on small arrays? | Thread creation overhead > actual sorting work |
| Why sequential merge beats parallel bubble? | O(n log n) fundamentally beats O(n²) regardless of threads |
| Why merge step is sequential? | Each merge decision depends on previous — no independent work |
| What does omp parallel for do? | Distributes loop iterations across thread team |
| What is odd phase in bubble sort? | Compare pairs at positions (1,2), (3,4), (5,6)... |
| What is even phase in bubble sort? | Compare pairs at positions (0,1), (2,3), (4,5)... |

---

*You've got this. The code works, the logic is sound, the benchmarks tell a clear story.*
*Key message for examiner: parallel helps, but algorithm choice matters more.*