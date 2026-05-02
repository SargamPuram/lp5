#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ============================
// Sequential Bubble Sort
// ============================
void bubbleSortSeq(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// ============================
// Parallel Bubble Sort (Odd-Even Transposition)
// ============================
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }

        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// ============================
// Merge Function
// ============================
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;

    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) arr[k++] = left[i++];
        else arr[k++] = right[j++];
    }

    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

// ============================
// Sequential Merge Sort
// ============================
void mergeSortSeq(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortSeq(arr, l, m);
        mergeSortSeq(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ============================
// Parallel Merge Sort
// ============================
void mergeSortParallel(vector<int>& arr, int l, int r, int depth) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Fallback to sequential to prevent thread explosion overhead
        if (depth <= 0) {
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
        merge(arr, l, m, r);
    }
}

void printArrayPreview(const vector<int>& arr) {
    int limit = 15; // How many elements to show in the preview
    int n = arr.size();
    
    for (int i = 0; i < min(limit, n); i++) {
        cout << arr[i] << " ";
    }
    if (n > limit) {
        cout << "... [" << n - limit << " more elements]";
    }
    cout << "\n";

    //  UNCOMMENT THE LINE BELOW IF THE EXAMINER WANTS TO SEE EVERY SINGLE ELEMENT
    // for (int x : arr) cout << x << " "; cout << "\n";
}
// ============================
// Main Execution & Benchmarking
// ============================
// ============================
// Main Execution & Benchmarking
// ============================
int main() {
    int size;
    cout << "Enter the number of elements to sort: ";
    cin >> size;

    vector<int> arr(size), temp;

    // Give control for manual input on small arrays
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
            for (int& x : arr) x = rand() % 10000;
        }
    } else {
        srand(time(0));
        for (int& x : arr) x = rand() % 10000;
        cout << "[Auto-generating " << size << " random elements for benchmarking...]\n";
    }

    cout << "\nUnsorted Array Preview: ";
    printArrayPreview(arr);
    cout << "\n--- Benchmarking Sorting Algorithms --- \n";

    // 1. Sequential Bubble Sort
    temp = arr;
    auto start = high_resolution_clock::now();
    bubbleSortSeq(temp);
    auto end = high_resolution_clock::now();
    cout << "[TIME] Sequential Bubble Sort: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // 2. Parallel Bubble Sort
    temp = arr;
    start = high_resolution_clock::now();
    bubbleSortParallel(temp);
    end = high_resolution_clock::now();
    cout << "[TIME] Parallel Bubble Sort:   " << duration_cast<milliseconds>(end - start).count() << " ms\n";
    
    // Verify Parallel Bubble
    cout << "   -> Parallel Bubble Verification: ";
    printArrayPreview(temp);

    // 3. Sequential Merge Sort
    temp = arr;
    start = high_resolution_clock::now();
    mergeSortSeq(temp, 0, size - 1);
    end = high_resolution_clock::now();
    cout << "[TIME] Sequential Merge Sort:  " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // 4. Parallel Merge Sort
    temp = arr;
    start = high_resolution_clock::now();
    mergeSortParallel(temp, 0, size - 1, 4); // Depth of 4 controls overhead
    end = high_resolution_clock::now();
    cout << "[TIME] Parallel Merge Sort:    " << duration_cast<milliseconds>(end - start).count() << " ms\n";
    
    // Verify Parallel Merge
    cout << "   -> Parallel Merge Verification:  ";
    printArrayPreview(temp);

    return 0;
}