# Parallel BFS & DFS using OpenMP — Complete Viva Guide
### LP5 | Group A | Assignment 1A & 1B

---

> **How to use this file:**
> Read each question, cover the answer, say it out loud.
> If you can say it confidently without reading — you're ready.

---

## SECTION 1 — Basic Concepts

---

### Q. What is BFS?

BFS stands for Breadth First Search. It is a graph traversal algorithm that visits all nodes level by level — first the start node, then all its direct neighbors, then their neighbors and so on. It uses a queue to track which nodes to visit next and a visited array to avoid revisiting nodes. BFS guarantees nodes are visited in order of their distance from the start node, which is why it is used to find shortest paths in unweighted graphs.

---

### Q. What is DFS?

DFS stands for Depth First Search. It is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It starts at a root node, goes deep into one branch completely, backtracks when no unvisited neighbors remain, then explores the next branch. DFS can be implemented using recursion where the function call stack acts as the stack, or iteratively using an explicit stack. It is used for cycle detection, topological sorting, finding connected components, and solving maze problems.

---

### Q. Difference between BFS and DFS?

| | BFS | DFS |
|---|---|---|
| Strategy | Level by level | Branch by branch |
| Data structure | Queue | Stack / Recursion |
| Finds shortest path | Yes (unweighted) | No |
| Memory usage | More (stores entire level) | Less (stores one path) |
| Use case | Shortest path, networking | Cycle detection, topological sort |

---

### Q. What data structure does BFS use and why?

BFS uses a queue because it needs to visit nodes in the order they were discovered — first in first out. Nodes at level 1 must all be processed before any node at level 2 is touched. A queue naturally enforces this ordering.

---

### Q. What data structure does DFS use and why?

DFS uses a stack because it needs to go as deep as possible before backtracking — last in first out. The recursive implementation uses the function call stack implicitly. The iterative version uses an explicit stack.

---

### Q. Why does our parallel BFS use a vector instead of a queue?

Two reasons. First, a queue has no level boundary — all levels get mixed together and you cannot tell where one level ends and the next begins. Our frontier vector represents exactly one level at a time. Second, omp for needs index-based access using an integer i to distribute iterations across threads. std::queue does not support index-based access. The frontier vector gives us both level control and index access, making it the correct structure for parallel BFS.

---

### Q. Why does our parallel DFS use recursion + tasks instead of an explicit stack?

The call stack of the recursion IS the DFS stack — it is still there. We just added omp task so each recursive branch can be picked up by a different thread. An explicit shared stack would need a lock around every push and pop, serializing all access and defeating parallelism. The task-based approach lets threads work on independent branches without contention.

---

## SECTION 2 — OpenMP Concepts

---

### Q. What is OpenMP?

OpenMP stands for Open Multi-Processing. It is an API that supports shared memory parallel programming in C, C++ and Fortran. It works using simple compiler directives called pragmas that we insert into existing sequential code to parallelize it without rewriting the entire program. It uses a fork-join model where a master thread forks worker threads for parallel regions and joins them back after. All threads share the same memory space. It is supported by all major compilers like GCC and Clang.

---

### Q. What is the fork-join model?

When the program hits omp parallel, the master thread forks into multiple worker threads. All threads execute the parallel block simultaneously. When the block ends, all threads join back into the master thread which continues sequentially. This can happen multiple times — each omp parallel is one fork-join cycle.

```
Master ──────────────fork──────────────────join──────► continues
                    /    \    \    \       /
               T1   T2   T3   T4  (work)
```

---

### Q. What are all the OpenMP directives and what do they do?

| Directive | What it does |
|---|---|
| `#pragma omp parallel` | Forks a team of threads, all execute the block |
| `#pragma omp for` | Splits loop iterations across threads |
| `#pragma omp parallel for` | Shorthand combining parallel + for |
| `#pragma omp critical` | Only one thread at a time executes this block |
| `#pragma omp atomic` | Lightweight single-operation thread safety |
| `#pragma omp task` | Creates independent work unit any thread can pick up |
| `#pragma omp taskwait` | Wait until all my child tasks finish |
| `#pragma omp single` | Only one thread runs this block, others wait |
| `#pragma omp sections` | Fixed number of parallel blocks known at compile time |
| `#pragma omp barrier` | All threads wait here until every thread arrives |
| `schedule(static)` | Pre-assigns equal fixed chunks to threads |
| `schedule(dynamic)` | Threads grab next available iteration when free |
| `firstprivate(var)` | Each task/thread gets its own copy initialized to current value |
| `private(var)` | Each thread gets its own uninitialized copy |
| `reduction(op:var)` | Safely combines each thread's result using operator |

---

### Q. What is a critical section?

A critical section is a block of code protected by omp critical where only one thread can execute at a time. All other threads wait outside until the current thread exits. It prevents race conditions on shared data. The cost is that it serializes access — too many critical sections reduce parallelism.

---

### Q. What is a race condition and how is it avoided?

A race condition occurs when two or more threads access shared data simultaneously and the final result depends on which thread executes first. Since thread scheduling is non-deterministic, the program gives different wrong results on different runs.

Example — two threads both check if a neighbor is unvisited, both see true, both mark it and both visit it. The node gets visited twice.

OpenMP solutions:
- **omp critical** — only one thread enters the block at a time
- **omp atomic** — for simple single operations like increment
- **firstprivate / private** — give each thread its own copy, no sharing
- **omp reduction** — safe combination of values across threads
- **omp barrier** — prevent threads from reading data before it is written

---

### Q. What is double check locking? Where did you use it?

Double check locking is a pattern with two checks on the same condition — one outside the critical section and one inside.

```cpp
if (!visited[nbr]) {               // outer check — avoids critical section
    #pragma omp critical
    {
        if (!visited[nbr]) {       // inner check — catches race between threads
            visited[nbr] = true;
            local_next.push_back(nbr);
        }
    }
}
```

The outer check filters out already-visited neighbors cheaply without acquiring the lock. The inner check catches the case where two threads both passed the outer check simultaneously — only the first one inside claims the neighbor. This reduces critical section contention significantly.

---

### Q. What is omp task?

omp task creates an independent unit of work that gets placed in a task pool. Any idle thread in the current team picks it up and executes it. Unlike omp for which needs a fixed loop, tasks are created dynamically at runtime. This makes tasks ideal for recursive algorithms like DFS where the number of branches is only known during execution.

---

### Q. What is omp taskwait?

omp taskwait is a synchronization point where the current thread waits until all tasks it directly spawned finish. Without taskwait in DFS, the recursive function would return immediately after spawning tasks without waiting for deeper branches to complete. The parallel region would end before the full DFS finishes, giving an incomplete traversal.

---

### Q. What is omp single and why do we use it for DFS?

omp single ensures exactly one thread executes a block while the others in the team wait at the end of it. We use it to make one thread initiate the DFS recursion.

Without omp single, all threads would simultaneously call parallelDFSUtil on the start node. All of them would push node 0 into dfs_order before any visited check can protect it — giving duplicate entries. With omp single, one thread starts the recursion cleanly. The other threads are not idle — they are alive and immediately pick up tasks as the recursion spawns them.

---

### Q. What is firstprivate and why did you use it on nbr?

firstprivate gives each task its own private copy of a variable initialized to the current value at the moment the task is created.

nbr is a loop variable that changes every iteration. Without firstprivate, all tasks would share the same nbr variable. By the time a task actually executes, the loop may have moved on to a different neighbor — all tasks would visit the wrong node.

firstprivate freezes the correct value of nbr for each task at the moment it is spawned.

```cpp
for (int nbr : adj[u]) {
    #pragma omp task firstprivate(nbr)   // each task gets its own nbr
    parallelDFSUtil(nbr, visited, dfs_order);
}
```

---

### Q. Why do you pass visited and dfs_order as pointers in DFS?

OpenMP tasks capture variables by value by default. A vector reference captured by value becomes a private copy — updates in child tasks are invisible to other tasks and the parent. A pointer captured by value is itself copied but still points to the same original object in memory. So all tasks across all recursion levels read and write the same visited array and dfs_order list.

```cpp
// WRONG — each task gets its own private copy of the vector
void parallelDFSUtil(int u, vector<bool>& visited, vector<int>& dfs_order)

// CORRECT — pointer is copied but still points to original shared object
void parallelDFSUtil(int u, vector<bool>* visited, vector<int>* dfs_order)
```

---

### Q. What is schedule(dynamic) and why did you use it in BFS?

schedule(dynamic) means threads do not get pre-assigned fixed chunks of loop iterations. Instead each thread picks up the next available iteration from a work queue as soon as it finishes its current one.

We use it in BFS because different frontier nodes can have very different numbers of neighbors. With static scheduling a thread assigned to a high-degree node would take much longer than others, leaving threads idle. Dynamic scheduling keeps all threads busy by load balancing automatically.

---

### Q. What is omp sections and when should you use it?

omp sections divides work into a fixed number of parallel blocks that are hardcoded at compile time. Each section runs on a different thread.

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    { taskA(); }

    #pragma omp section
    { taskB(); }
}
```

Use sections when you have a small fixed number of independent tasks known at compile time — like processing left and right subtrees of a balanced binary tree. Do not use for DFS on general graphs because the number of branches depends on the graph structure and is only known at runtime.

---

### Q. Why did you use tasks and not sections for DFS?

Sections require a fixed number of parallel blocks hardcoded at compile time. DFS branches equal the number of unvisited neighbors of each node — unknown until runtime. You cannot write a sections block that handles 1 neighbor for one node and 5 neighbors for another dynamically.

Tasks are created inside a loop — one task per unvisited neighbor regardless of how many there are. Additionally tasks reuse one thread pool created by a single omp parallel, while sections re-enter omp parallel at every recursive level paying thread creation cost every time. Our benchmark proved this — task-based DFS took 148 microseconds, sections-based took 7968 microseconds on the same graph.

---

## SECTION 3 — Parallel Algorithm Design

---

### Q. How does parallel BFS work?

Parallel BFS processes each level of the graph in parallel. All nodes at the same level are independent of each other so they can be processed simultaneously.

1. Initialize frontier vector with start node
2. While frontier is not empty:
   - Record current level into output (sequential, for determinism)
   - Open omp parallel region — fork thread team
   - Each thread gets a private local_next vector
   - omp for with dynamic scheduling distributes frontier nodes across threads
   - Each thread checks neighbors of its assigned nodes
   - Unvisited neighbors are claimed inside critical section using double check locking
   - Each thread merges its local_next into shared next_frontier under critical section
   - Parallel region ends — threads join
   - frontier = next_frontier, repeat

---

### Q. How does parallel DFS work?

Parallel DFS explores independent branches of the DFS tree simultaneously using tasks.

1. Initialize visited array and dfs_order list
2. Open one omp parallel region — thread pool created once
3. omp single makes one thread call the recursive utility
4. Inside utility — record current node under critical section
5. For each neighbor — check and claim inside critical section, mark visited before spawning task
6. Spawn omp task with firstprivate(nbr) for each claimed neighbor
7. omp taskwait — wait for all child tasks to finish before returning
8. Other threads in the pool pick up tasks as they are spawned
9. Parallel region ends only after all tasks complete

---

### Q. What is the time complexity of parallel BFS and DFS?

| | Sequential | Parallel |
|---|---|---|
| BFS | O(V + E) | O((V + E) / p) per level |
| DFS | O(V + E) | O((V + E) / p) with tasks |

V = vertices, E = edges, p = number of threads.
Gains are most visible on large dense graphs. On small graphs thread overhead dominates.

---

### Q. What is the space complexity?

| | Space |
|---|---|
| BFS | O(V) for visited + frontier vectors |
| DFS | O(V) for visited + recursion call stack depth |

---

### Q. Why is parallel BFS slower than parallel DFS on small graphs?

BFS enters omp parallel once per level. Our graph has 3 levels so the thread pool is created and destroyed 3 times. Each fork-join on WSL costs thousands of microseconds in thread creation overhead.

DFS enters omp parallel exactly once and uses lightweight tasks inside that single pool. On a small graph the actual graph work takes negligible time so overhead dominates — and BFS has more overhead.

On large graphs with thousands of nodes per level, BFS parallelism pays off because per-level work justifies the fork-join cost.

---

### Q. What are the applications of parallel BFS?

- Shortest path finding in maps and navigation — Google Maps
- Social network analysis — degrees of separation
- Web crawling — search engines discover pages level by level
- Network broadcasting — finding all reachable nodes
- Peer to peer networks — resource discovery
- Game AI — pathfinding for characters
- Garbage collection — tracing reachable objects in memory

---

### Q. What are the advantages of parallel DFS?

- Multiple branches explored simultaneously — speedup on large graphs
- Better CPU utilization — all cores kept busy
- Independent subtrees have no dependency — naturally parallel
- Scalable — performance improves with more cores
- Reduced wall clock time even though total work is same

---

## SECTION 4 — Tricky Examiner Questions

---

### Q. Your manual says queue is used in BFS but your code uses vectors. Is your code wrong?

No. The manual describes the concept where each thread maintains a local queue — our local_next vector serves exactly that purpose. Each thread collects discovered nodes locally and merges at the end. The underlying idea is identical, just implemented with vectors because omp for requires index-based access that std::queue does not support. The frontier vector is the parallel-safe equivalent of the BFS queue.

---

### Q. Your manual mentions parallel for for DFS. Why did you use tasks?

The manual gives a general description. omp parallel for only works on loops with a known fixed iteration count. DFS recursion does not have a fixed loop — the depth and branching factor depend entirely on the graph. omp task is the correct OpenMP primitive for dynamic recursion. Tasks were introduced specifically in OpenMP 3.0 to handle exactly this kind of irregular parallel work.

---

### Q. Can parallel DFS give a different traversal order than sequential DFS?

Yes and this is expected. Sequential DFS always produces the same order because one thread follows one path. Parallel DFS order can vary between runs because multiple threads explore different branches simultaneously and whichever thread finishes first records its nodes first. Both visit all nodes correctly — only the order differs. This is acceptable behavior in parallel graph traversal.

---

### Q. What happens if you remove omp taskwait?

Without taskwait, parallelDFSUtil returns immediately after spawning tasks for its neighbors without waiting for those tasks to complete. The call stack unwinds, the parallel region can end, and deeper branches that haven't finished yet get cut off. The result is an incomplete traversal — some nodes never appear in the output.

---

### Q. What happens if you remove omp single and let all threads start DFS?

All threads would simultaneously call parallelDFSUtil on the start node. All of them would push node 0 into dfs_order before any critical section can protect it — giving node 0 recorded multiple times. Then all threads would race to claim the same neighbors, and even though the critical section prevents duplicate neighbor visits, the start node itself has no protection at that point. omp single gives one controlled entry point while keeping all threads alive and ready for tasks.

---

### Q. What is the difference between omp atomic and omp critical?

| | omp atomic | omp critical |
|---|---|---|
| Scope | Single simple operation only | Any block of code |
| Operations | ++, --, +=, -=, =  | Anything |
| Overhead | Very low | Higher |
| Use when | Incrementing a counter | Checking + modifying + inserting |

We use critical not atomic because our protected operations involve a condition check, an assignment, and a vector push — three operations that must happen together atomically. Atomic only protects one operation at a time.

---

### Q. What is shared memory and why is it important for OpenMP?

Shared memory means all threads in a process access the same physical RAM. There is no need to copy data between threads — they all read and write the same variables directly. This is why OpenMP is efficient — communication between threads is just a memory read, not a network message or inter-process call. The tradeoff is that shared access to the same data requires synchronization via critical sections to prevent race conditions.

---

### Q. How many threads does OpenMP use by default?

By default OpenMP uses as many threads as there are logical CPU cores on the machine. This can be controlled with:
- Environment variable: `export OMP_NUM_THREADS=4`
- In code: `omp_set_num_threads(4)`
- Directive clause: `#pragma omp parallel num_threads(4)`

---

### Q. What is the compile command for OpenMP?

```bash
g++ -fopenmp filename.cpp -o outputname
```

The `-fopenmp` flag tells GCC to recognize and process all `#pragma omp` directives. Without it, all pragmas are silently ignored and the code runs sequentially.

---

## SECTION 5 — Quick Revision Flashcards

| Question | One line answer |
|---|---|
| What is BFS? | Level by level traversal using queue |
| What is DFS? | Branch by branch traversal using stack/recursion |
| What is OpenMP? | API for shared memory parallel programming using pragmas |
| What is fork-join? | Master forks threads, all work, all join back |
| What is omp parallel? | Creates team of threads |
| What is omp for? | Splits loop iterations across threads |
| What is omp critical? | Only one thread at a time |
| What is omp task? | Dynamic work unit any thread can pick up |
| What is omp taskwait? | Wait for all my child tasks |
| What is omp single? | Only one thread runs this |
| What is firstprivate? | Each task gets own copy initialized to current value |
| What is race condition? | Result depends on thread execution order — wrong output |
| What is double check locking? | Check outside critical + check again inside |
| Why vector not queue in BFS? | Index access for omp for + level boundary control |
| Why tasks not sections for DFS? | Branches are dynamic — unknown at compile time |
| Why pointers not references in DFS? | Tasks capture by value — pointer still points to original |
| Why omp single for DFS start? | Prevent all threads visiting start node simultaneously |
| Why dynamic schedule in BFS? | Nodes have unequal neighbor counts — load balance |
| Why BFS slower on small graph? | Fork-join cost per level dominates actual graph work |

---

*Good luck! You know this — trust the code, trust the explanation.*