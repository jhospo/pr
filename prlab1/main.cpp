#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;

const int MAX = 100;

void multiplyMatrixSequential(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiplyMatrixParallel(double* A, double* B, double* C, int n) {
//#pragma omp parallel
//    {
//        int thread_num = omp_get_thread_num();
//        int total_threads = omp_get_num_threads();
//
//#pragma omp single
//        {
//            cout << "threads: " << total_threads << "\n";
//        }
//    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}


int main() {
    const bool USE_IO = true;
    const bool USE_PARALLEL = true;
    omp_set_num_threads(4);
    int n;
    double* A;
    double* B;
    double* C;

    if (USE_IO) {
        ifstream fin("macierz.txt");
        ofstream fout("wynik.txt");

        if (!fin || !fout) {
            cout << "wrong file" << endl;
            return 1;
        }

        int n1, n2;
        fin >> n1 >> n2;

        if (n1 != n2 || n1 <= 0 || n1 > MAX) {
            cout << "wrong size of matrix" << endl;
            return 1;
        }

        n = n1;

        A = new double[n * n];
        B = new double[n * n];
        C = new double[n * n];

        for (int i = 0; i < n * n; i++) fin >> A[i];
        for (int i = 0; i < n * n; i++) fin >> B[i];

        if (USE_PARALLEL)
            multiplyMatrixParallel(A, B, C, n);
        else
            multiplyMatrixSequential(A, B, C, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fout << C[i * n + j] << " ";
            }
            fout << "\n";
        }

        fin.close();
        fout.close();
    } else {
        n = 4;
        A = new double[n * n];
        B = new double[n * n];
        C = new double[n * n];

        for (int i = 0; i < n * n; i++) A[i] = 1.0;
        for (int i = 0; i < n * n; i++) B[i] = 2.0;

        if (USE_PARALLEL)
            multiplyMatrixParallel(A, B, C, n);
        else
            multiplyMatrixSequential(A, B, C, n);

        cout << "Wynik:\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << C[i * n + j] << " ";
            }
            cout << "\n";
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
