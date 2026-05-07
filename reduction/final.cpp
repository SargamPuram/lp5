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
// Sequential Operations (Decoupled)
// ============================
void seqMin(const vector<int>& arr) {
    int minVal = arr[0];
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < arr.size(); i++)
        if (arr[i] < minVal) minVal = arr[i];
    auto end = high_resolution_clock::now();
    cout << "[Sequential] Min value: " << minVal
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

void seqMax(const vector<int>& arr) {
    int maxVal = arr[0];
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < arr.size(); i++)
        if (arr[i] > maxVal) maxVal = arr[i];
    auto end = high_resolution_clock::now();
    cout << "[Sequential] Max value: " << maxVal
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

void seqSum(const vector<int>& arr) {
    long long sum = 0;
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < arr.size(); i++)
        sum += arr[i];
    auto end = high_resolution_clock::now();
    cout << "[Sequential] Sum:       " << sum
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

void seqAvg(const vector<int>& arr) {
    long long sum = 0;
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < arr.size(); i++)
        sum += arr[i];
    double avg = (double)sum / arr.size();
    auto end = high_resolution_clock::now();
    cout << "[Sequential] Average:   " << avg
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

// ============================
// Parallel Operations (Decoupled)
// ============================
void parMin(const vector<int>& arr) {
    int minValue = arr[0];
    auto start = high_resolution_clock::now();
    #pragma omp parallel for reduction(min : minValue)
    for (size_t i = 0; i < arr.size(); i++)
        if (arr[i] < minValue) minValue = arr[i];
    auto end = high_resolution_clock::now();
    cout << "[Parallel]   Min value: " << minValue
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

void parMax(const vector<int>& arr) {
    int maxValue = arr[0];
    auto start = high_resolution_clock::now();
    #pragma omp parallel for reduction(max : maxValue)
    for (size_t i = 0; i < arr.size(); i++)
        if (arr[i] > maxValue) maxValue = arr[i];
    auto end = high_resolution_clock::now();
    cout << "[Parallel]   Max value: " << maxValue
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

void parSum(const vector<int>& arr) {
    long long sum = 0;
    auto start = high_resolution_clock::now();
    #pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < arr.size(); i++)
        sum += arr[i];
    auto end = high_resolution_clock::now();
    cout << "[Parallel]   Sum:       " << sum
         << " (Time: " << duration_cast<microseconds>(end - start).count() << " us)\n";
}

void parAvg(const vector<int>& arr) {
    long long sum = 0;
    auto start = high_resolution_clock::now();
    #pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < arr.size(); i++)
        sum += arr[i];
    double avg = (double)sum / arr.size();
    auto end = high_resolution_clock::now();
    cout << "[Parallel]   Average:   " << avg
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
            for (int i = 0; i < size; i++) cin >> arr[i];
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

    // Warm up OpenMP thread pool before benchmarking
    #pragma omp parallel for
    for (int i = 0; i < 1000; i++) {}

    cout << "-------------------------------------------------\n";
    // Sequential decoupled
    cout << "Sequential Operations:\n";
    auto s1 = high_resolution_clock::now();
    seqMin(arr);
    seqMax(arr);
    seqSum(arr);
    seqAvg(arr);
    auto s2 = high_resolution_clock::now();
    cout << "  >> Total Sequential Time: "
         << duration_cast<microseconds>(s2 - s1).count() << " us\n";

    cout << "-------------------------------------------------\n";

    // Parallel decoupled
    cout << "Parallel Operations:\n";
    auto p1 = high_resolution_clock::now();
    parMin(arr);
    parMax(arr);
    parSum(arr);
    parAvg(arr);
    auto p2 = high_resolution_clock::now();
    cout << "  >> Total Parallel Time:     "
         << duration_cast<microseconds>(p2 - p1).count() << " us\n";

    cout << "-------------------------------------------------\n";

    return 0;
}