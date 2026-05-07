#include <iostream>
#include <vector>
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
        adj[v].push_back(u);
    }

    // =========================================================
    // PARALLEL BFS — with search
    // =========================================================
    void parallelBFS(int start, int target = -1) {
        vector<bool> visited(V, false);
        vector<int> frontier;
        vector<int> bfs_order;
        bool found = false;

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
                                    if (nbr == target) found = true;
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

        cout << "\nParallel BFS Traversal: ";
        for (int node : bfs_order)
            cout << node << " ";
        cout << endl;

        if (target != -1) {
            if (found || target == start)
                cout << "[SEARCH] Node " << target << " FOUND via BFS.\n";
            else
                cout << "[SEARCH] Node " << target << " NOT found via BFS.\n";
        }
    }

    // =========================================================
    // PARALLEL DFS — shared stack, with search
    // =========================================================
    void parallelDFS(int start, int target = -1) {
        vector<bool> visited(V, false);
        vector<int> dfs_order;
        stack<int> stk;
        bool found = false;

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
                        if (u == target) found = true;
                    }
                }

                if (u == -1) break;

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

        cout << "\nParallel DFS Traversal: ";
        for (int node : dfs_order)
            cout << node << " ";
        cout << endl;

        if (target != -1) {
            if (found || target == start)
                cout << "[SEARCH] Node " << target << " FOUND via DFS.\n";
            else
                cout << "[SEARCH] Node " << target << " NOT found via DFS.\n";
        }
    }
};

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

    int target;
    cout << "Enter node to search (-1 to skip): ";
    cin >> target;

    // BFS
    auto t1 = high_resolution_clock::now();
    g.parallelBFS(start, target);
    auto t2 = high_resolution_clock::now();
    cout << "[TIME] BFS: "
         << duration_cast<microseconds>(t2 - t1).count()
         << " microseconds\n";

    // DFS
    auto t3 = high_resolution_clock::now();
    g.parallelDFS(start, target);
    auto t4 = high_resolution_clock::now();
    cout << "[TIME] DFS: "
         << duration_cast<microseconds>(t4 - t3).count()
         << " microseconds\n";

    return 0;
}