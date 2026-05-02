#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// NOTE ON THIS FILE:
// This is a BACKUP version using #pragma omp sections instead of omp task.
// Sections are best for a fixed number of independent parallel blocks.
// For DFS, tasks are the correct approach (see bfs_dfs.cpp).
// This file exists in case the lab system has issues with omp task.

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
    // BFS using omp sections
    // Splits each frontier level into two halves processed in parallel
    // =========================================================
    void sectionsBFS(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;
        vector<int> bfs_order;

        visited[start] = true;
        frontier.push_back(start);

        while (!frontier.empty()) {
            for (int node : frontier)
                bfs_order.push_back(node);

            vector<int> next_left, next_right;
            size_t mid = frontier.size() / 2;

            // Split frontier into two halves — each section handles one half
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    // Left half of the frontier
                    for (size_t i = 0; i < mid; i++) {
                        int u = frontier[i];
                        for (int nbr : adj[u]) {
                            if (!visited[nbr]) {
                                #pragma omp critical
                                {
                                    if (!visited[nbr]) {
                                        visited[nbr] = true;
                                        next_left.push_back(nbr);
                                    }
                                }
                            }
                        }
                    }
                }

                #pragma omp section
                {
                    // Right half of the frontier
                    for (size_t i = mid; i < frontier.size(); i++) {
                        int u = frontier[i];
                        for (int nbr : adj[u]) {
                            if (!visited[nbr]) {
                                #pragma omp critical
                                {
                                    if (!visited[nbr]) {
                                        visited[nbr] = true;
                                        next_right.push_back(nbr);
                                    }
                                }
                            }
                        }
                    }
                }
            } // implicit barrier — both sections finish before continuing

            // Merge the two halves
            frontier = next_left;
            frontier.insert(frontier.end(), next_right.begin(), next_right.end());
        }

        cout << "\n[SECTIONS] BFS Traversal: ";
        for (int node : bfs_order)
            cout << node << " ";
        cout << endl;
    }

    // =========================================================
    // DFS using omp sections
    // NOTE: sections only parallelize exactly 2 branches.
    // Real graphs can have any number of neighbors, so this
    // is a demonstration of sections syntax — not full parallelism.
    // Use bfs_dfs.cpp (omp task version) for correct parallel DFS.
    // =========================================================
    void sectionsDFSUtil(int u, vector<bool>& visited, vector<int>& dfs_order) {
        #pragma omp critical
        {
            visited[u] = true;
            dfs_order.push_back(u);
        }

        vector<int> unvisited_neighbors;

        // Collect unvisited neighbors first (outside critical)
        for (int nbr : adj[u]) {
            bool claim = false;
            #pragma omp critical
            {
                if (!visited[nbr]) {
                    visited[nbr] = true; // Claim before spawning
                    claim = true;
                }
            }
            if (claim)
                unvisited_neighbors.push_back(nbr);
        }

        // Parallelize first two branches using sections
        // Any branches beyond 2 are handled sequentially after
        if (unvisited_neighbors.size() >= 2) {
            #pragma omp parallel sections
            {
                #pragma omp section
                sectionsDFSUtil(unvisited_neighbors[0], visited, dfs_order);

                #pragma omp section
                sectionsDFSUtil(unvisited_neighbors[1], visited, dfs_order);
            }
            // Handle remaining neighbors sequentially
            for (size_t i = 2; i < unvisited_neighbors.size(); i++)
                sectionsDFSUtil(unvisited_neighbors[i], visited, dfs_order);

        } else {
            // 0 or 1 neighbor — no parallelism possible, just recurse
            for (int nbr : unvisited_neighbors)
                sectionsDFSUtil(nbr, visited, dfs_order);
        }
    }

    void sectionsDFS(int start) {
        vector<bool> visited(V, false);
        vector<int> dfs_order;

        sectionsDFSUtil(start, visited, dfs_order);

        cout << "\n[SECTIONS] DFS Traversal: ";
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
    g.sectionsBFS(start);
    auto t2 = high_resolution_clock::now();
    cout << "[TIME] BFS: "
         << duration_cast<microseconds>(t2 - t1).count()
         << " microseconds\n";

    // DFS benchmark
    auto t3 = high_resolution_clock::now();
    g.sectionsDFS(start);
    auto t4 = high_resolution_clock::now();
    cout << "[TIME] DFS: "
         << duration_cast<microseconds>(t4 - t3).count()
         << " microseconds\n";

    return 0;
}