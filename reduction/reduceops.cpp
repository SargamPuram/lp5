#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

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
// Sequential Baseline (For Proof)
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
    cout << "[Sequential Baseline]\n";
    cout << "  Min: " << min_val << " | Max: " << max_val 
         << " | Sum: " << sum << " | Avg: " << avg << "\n";
    cout << "  -> Total Sequential Time: " << duration_cast<microseconds>(end - start).count() << " us\n";
}

// ============================
// 1. Parallel Minimum Operation
// ============================
void minOperation(const vector<int>& arr) {
    int minValue = arr[0];
    auto start = high_resolution_clock::now();

    #pragma omp parallel for reduction(min : minValue)
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] < minValue) {
            minValue = arr[i];
        }
    }

    auto end = high_resolution_clock::now();
    cout << "[Parallel] Min value: " << minValue 
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

// ============================
// 2. Parallel Maximum Operation
// ============================
void maxOperation(const vector<int>& arr) {
    int maxValue = arr[0];
    auto start = high_resolution_clock::now();

    #pragma omp parallel for reduction(max : maxValue)
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] > maxValue) {
            maxValue = arr[i];
        }
    }

    auto end = high_resolution_clock::now();
    cout << "[Parallel] Max value: " << maxValue 
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

// ============================
// 3. Parallel Sum Operation
// ============================
void sumOperation(const vector<int>& arr) {
    long long sum = 0; 
    auto start = high_resolution_clock::now();

    #pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }

    auto end = high_resolution_clock::now();
    cout << "[Parallel] Sum:       " << sum 
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

// ============================
// 4. Parallel Average Operation
// ============================
void avgOperation(const vector<int>& arr) {
    long long sum = 0;
    auto start = high_resolution_clock::now();

    #pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    
    double avg = (double)sum / arr.size();

    auto end = high_resolution_clock::now();
    cout << "[Parallel] Average:   " << avg 
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
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
    
    // Run the sequential baseline first
    seqOperations(arr);

    cout << "-------------------------------------------------\n";
    cout << "Executing Decoupled Parallel Operations...\n\n";

    // Run the isolated parallel operations
    minOperation(arr);
    maxOperation(arr);
    sumOperation(arr);
    avgOperation(arr);

    cout << "\n-------------------------------------------------\n";

    return 0;
}