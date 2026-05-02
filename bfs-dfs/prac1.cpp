#include <iostream>
#include <vector>
#include <queue>
#include <stack>
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
        adj[v].push_back(u); // Undirected graph
    }

    // =========================================================
    // SEQUENTIAL BFS
    // =========================================================
    void sequentialBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "\n[SEQ] BFS Traversal: ";

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            cout << u << " ";

            for (int nbr : adj[u]) {
                if (!visited[nbr]) {
                    visited[nbr] = true;
                    q.push(nbr);
                }
            }
        }
        cout << endl;
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

                #pragma omp critical
                next_frontier.insert(next_frontier.end(),
                                     local_next.begin(), local_next.end());
            }

            frontier = next_frontier;
        }

        cout << "\n[PAR] BFS Traversal: ";
        for (int node : bfs_order)
            cout << node << " ";
        cout << endl;
    }

    // =========================================================
    // SEQUENTIAL DFS
    // =========================================================
    void sequentialDFSUtil(int u, vector<bool>& visited) {
        visited[u] = true;
        cout << u << " ";

        for (int nbr : adj[u]) {
            if (!visited[nbr])
                sequentialDFSUtil(nbr, visited);
        }
    }

    void sequentialDFS(int start) {
        vector<bool> visited(V, false);
        cout << "\n[SEQ] DFS Traversal: ";
        sequentialDFSUtil(start, visited);
        cout << endl;
    }

    // =========================================================
    // PARALLEL DFS — shared stack approach (WSL-safe)
    // Multiple threads pull nodes from a shared stack and
    // push unvisited neighbors back. Critical sections protect
    // the shared stack and visited array from race conditions.
    // =========================================================
    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        vector<int> dfs_order;
        stack<int> stk;

        visited[start] = true;
        stk.push(start);

        #pragma omp parallel
        {
            while (true) {
                int u = -1;

                #pragma omp critical
                {
                    if (!stk.empty()) {
                        u = stk.top();
                        stk.pop();
                        dfs_order.push_back(u);
                    }
                }

                if (u == -1) break; // stack empty, thread exits

                for (int nbr : adj[u]) {
                    #pragma omp critical
                    {
                        if (!visited[nbr]) {
                            visited[nbr] = true;
                            stk.push(nbr);
                        }
                    }
                }
            }
        }

        cout << "\n[PAR] DFS Traversal: ";
        for (int node : dfs_order)
            cout << node << " ";
        cout << endl;
    }
};

// =========================================================
// BENCHMARK HELPER
// =========================================================
void printBenchmark(const string& label, long long seq_us, long long par_us) {
    cout << "\n--- " << label << " Benchmark ---\n";
    cout << "  Sequential : " << seq_us << " microseconds\n";
    cout << "  Parallel   : " << par_us << " microseconds\n";

    if (par_us > 0) {
        double speedup = (double)seq_us / par_us;
        cout << "  Speedup    : " << speedup << "x\n";
        if (speedup < 1.0)
            cout << "  Note: Parallel overhead > gain on small graphs. "
                    "Try a larger graph for meaningful speedup.\n";
    }
}

// =========================================================
// MAIN
// =========================================================
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

    cout << "\n========== BFS ==========";

    auto t1 = high_resolution_clock::now();
    g.sequentialBFS(start);
    auto t2 = high_resolution_clock::now();
    long long seq_bfs = duration_cast<microseconds>(t2 - t1).count();

    auto t3 = high_resolution_clock::now();
    g.parallelBFS(start);
    auto t4 = high_resolution_clock::now();
    long long par_bfs = duration_cast<microseconds>(t4 - t3).count();

    printBenchmark("BFS", seq_bfs, par_bfs);

    cout << "\n========== DFS ==========";

    auto t5 = high_resolution_clock::now();
    g.sequentialDFS(start);
    auto t6 = high_resolution_clock::now();
    long long seq_dfs = duration_cast<microseconds>(t6 - t5).count();

    auto t7 = high_resolution_clock::now();
    g.parallelDFS(start);
    auto t8 = high_resolution_clock::now();
    long long par_dfs = duration_cast<microseconds>(t8 - t7).count();

    printBenchmark("DFS", seq_dfs, par_dfs);

    return 0;
}