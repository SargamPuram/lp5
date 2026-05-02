#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // =========================================================
    // PARALLEL BFS
    // =========================================================
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;
        vector<int> bfs_order;

        visited[start] = true;
        frontier.push_back(start);

        while (!frontier.empty()) {
            vector<int> next_frontier;

            // Record current level — done outside parallel region
            // so output order is always deterministic
            for (int node : frontier)
                bfs_order.push_back(node);

            #pragma omp parallel
            {
                vector<int> local_next;

                #pragma omp for schedule(dynamic)
                for (size_t i = 0; i < frontier.size(); i++) {
                    int u = frontier[i];
                    for (int nbr : adj[u]) {
                        if (!visited[nbr]) {
                            // Double-check locking: outer check avoids
                            // entering critical every time, inner check
                            // prevents duplicate insertions from racing threads
                            #pragma omp critical
                            {
                                if (!visited[nbr]) {
                                    visited[nbr] = true;
                                    local_next.push_back(nbr);
                                }
                            }
                        }
                    }
                }

                // Merge each thread's local results into shared frontier
                #pragma omp critical
                next_frontier.insert(next_frontier.end(),
                                     local_next.begin(), local_next.end());
            }

            frontier = next_frontier;
        }

        cout << "\nParallel BFS Traversal: ";
        for (int node : bfs_order)
            cout << node << " ";
        cout << endl;
    }

    // =========================================================
    // PARALLEL DFS
    // =========================================================
    // visited and dfs_order are passed by POINTER so that
    // OpenMP tasks (which capture by value by default) still
    // refer to the same shared objects across all task invocations
    void parallelDFSUtil(int u, vector<bool>* visited,
                         vector<int>* dfs_order) {
        #pragma omp critical
        dfs_order->push_back(u);

        for (int nbr : adj[u]) {
            bool should_visit = false;

            // Claim the neighbor atomically BEFORE spawning the task
            // If we checked inside the task it would be too late —
            // two threads could both pass the check and visit the same node
            #pragma omp critical
            {
                if (!(*visited)[nbr]) {
                    (*visited)[nbr] = true;
                    should_visit = true;
                }
            }

            if (should_visit) {
                // Capture nbr by value (its own copy per task),
                // but visited/dfs_order by pointer so all tasks
                // share the same data structures
                #pragma omp task firstprivate(nbr)
                parallelDFSUtil(nbr, visited, dfs_order);
            }
        }

        // Wait for ALL child tasks spawned here before returning
        // Without this, the parallel region could end before
        // deeper recursion levels finish
        #pragma omp taskwait
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        vector<int> dfs_order;

        #pragma omp parallel
        {
            #pragma omp single
            {
                // Mark and record start node here, inside the parallel region,
                // so the task team is already active when recursion begins
                visited[start] = true;
                parallelDFSUtil(start, &visited, &dfs_order);
            }
        }

        cout << "\nParallel DFS Traversal: ";
        for (int node : dfs_order)
            cout << node << " ";
        cout << endl;
    }
};

int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;

    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting vertex: ";
    cin >> start;

    // BFS benchmark
    auto t1 = high_resolution_clock::now();
    g.parallelBFS(start);
    auto t2 = high_resolution_clock::now();
    cout << "[TIME] BFS: "
         << duration_cast<microseconds>(t2 - t1).count()
         << " microseconds\n";

    // DFS benchmark
    auto t3 = high_resolution_clock::now();
    g.parallelDFS(start);
    auto t4 = high_resolution_clock::now();
    cout << "[TIME] DFS: "
         << duration_cast<microseconds>(t4 - t3).count()
         << " microseconds\n";

    return 0;
}