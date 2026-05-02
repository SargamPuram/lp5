#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <climits>

using namespace std;
using namespace std::chrono;

// ============================
// Array Preview Utility
// ============================
void printArrayPreview(const vector<int>& arr) {
    int limit = 15;
    int n = arr.size();
    for (int i = 0; i < min(limit, n); i++) {
        cout << arr[i] << " ";
    }
    if (n > limit) cout << "... [" << n - limit << " more elements]";
    cout << "\n";
}

// ============================
// Sequential Operations
// ============================
void seqOperations(const vector<int>& arr) {
    int n = arr.size();
    int min_val = arr[0];
    int max_val = arr[0];
    long long sum = 0;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
        sum += arr[i];
    }
    double avg = (double)sum / n;

    auto end = high_resolution_clock::now();
    cout << "[Sequential] Min: " << min_val << ", Max: " << max_val 
         << ", Sum: " << sum << ", Avg: " << avg << "\n";
    cout << "  -> Time taken: " << duration_cast<microseconds>(end - start).count() << " microseconds\n\n";
}

// ============================
// Parallel Reduction Operations
// ============================
void parallelOperations(const vector<int>& arr) {
    int n = arr.size();
    int min_val = arr[0];
    int max_val = arr[0];
    long long sum = 0;

    auto start = high_resolution_clock::now();

    // The reduction clause automatically handles thread-safety and merges the results
    #pragma omp parallel for reduction(min: min_val) reduction(max: max_val) reduction(+: sum)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
        sum += arr[i];
    }
    double avg = (double)sum / n;

    auto end = high_resolution_clock::now();
    cout << "[Parallel]   Min: " << min_val << ", Max: " << max_val 
         << ", Sum: " << sum << ", Avg: " << avg << "\n";
    cout << "  -> Time taken: " << duration_cast<microseconds>(end - start).count() << " microseconds\n\n";
}

// ============================
// Main Execution
// ============================
int main() {
    int size;
    cout << "Enter the number of elements: ";
    cin >> size;

    vector<int> arr(size);

    if (size <= 20) {
        char choice;
        cout << "Do you want to enter elements manually? (y/n): ";
        cin >> choice;
        
        if (choice == 'y' || choice == 'Y') {
            cout << "Enter " << size << " elements: ";
            for (int i = 0; i < size; i++) {
                cin >> arr[i];
            }
        } else {
            srand(time(0));
            for (int& x : arr) x = rand() % 1000;
        }
    } else {
        srand(time(0));
        for (int& x : arr) x = rand() % 1000;
        cout << "[Auto-generating " << size << " random elements...]\n";
    }

    cout << "\nArray Preview: ";
    printArrayPreview(arr);
    cout << "\n-------------------------------------------------\n";

    seqOperations(arr);
    parallelOperations(arr);

    return 0;
}