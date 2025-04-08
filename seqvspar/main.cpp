#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <string>
#include <vector>
#include <random>

struct SchedResult {
    int n;
    int threads;
    std::string sched;
    int chunk;
    double time;
};
void generateMatrices(int n, const std::string &fileA, const std::string &fileB) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10);

    {
        std::ofstream fout(fileA);
        fout << n << "\n";
        for (int i = 0; i < n * n; i++) {
            fout << dist(gen) << ((i + 1) % n == 0 ? '\n' : ' ');
        }
    }
    {
        std::ofstream fout(fileB);
        fout << n << "\n";
        for (int i = 0; i < n * n; i++) {
            fout << dist(gen) << ((i + 1) % n == 0 ? '\n' : ' ');
        }
    }
    std::cout << "Generated random matrices (" << n << "x" << n << ") in "
              << fileA << " and " << fileB << "\n";
}

void readMatrixOneDim(const std::string &filename, double *arr, int &n) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    fin >> n;
    for (int i = 0; i < n * n; i++) {
        fin >> arr[i];
    }
    fin.close();
}

void writeMatrixOneDim(const std::string &filename, const double *arr, int n) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    fout << n << "\n";
    for (int i = 0; i < n * n; i++) {
        fout << arr[i] << (((i + 1) % n == 0) ? '\n' : ' ');
    }
    fout.close();
}

void multiplySequential(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double sum = 0.0;
            for (int k = 0; k < n; k++){
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiplyParallelI(const double *A, const double *B, double *C, int n, int threads) {
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double sum = 0.0;
            for (int k = 0; k < n; k++){
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiplyParallelJ(const double *A, const double *B, double *C, int n, int threads) {
    for (int i = 0; i < n; i++){
#pragma omp parallel for num_threads(threads)
        for (int j = 0; j < n; j++){
            double sum = 0.0;
            for (int k = 0; k < n; k++){
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiplyParallelK(const double *A, const double *B, double *C, int n, int threads) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double sum = 0.0;
#pragma omp parallel for num_threads(threads)
            for (int k = 0; k < n; k++){
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
int main(){
    std::vector<int> sizes = {100, 500, 1000, 2000};
    std::vector<int> parThreads = {2, 4, 8, 16};

    for (int n : sizes) {
        generateMatrices(n, "A_" + std::to_string(n) + ".txt",
                         "B_" + std::to_string(n) + ".txt");
    }

    std::vector<double> seqTimes(sizes.size(), 0.0);
    std::vector<std::vector<double>> iTimes(sizes.size(), std::vector<double>(parThreads.size(), 0.0));
    std::vector<std::vector<double>> jTimes(sizes.size(), std::vector<double>(parThreads.size(), 0.0));
    std::vector<std::vector<double>> kTimes(sizes.size(), std::vector<double>(parThreads.size(), 0.0));

    for (size_t si = 0; si < sizes.size(); si++){
        int n = sizes[si];

        double *A = new double[n * n];
        double *B = new double[n * n];
        double *C = new double[n * n];

        {
            std::string fa = "A_" + std::to_string(n) + ".txt";
            std::string fb = "B_" + std::to_string(n) + ".txt";
            int nA = 0, nB = 0;

            auto startReadParallel = std::chrono::high_resolution_clock::now();
#pragma omp parallel sections
            {
#pragma omp section
                {
                    readMatrixOneDim(fa, A, nA);
                }
#pragma omp section
                {
                    readMatrixOneDim(fb, B, nB);
                }
            }
            auto endReadParallel = std::chrono::high_resolution_clock::now();
            double parallelReadTime = std::chrono::duration<double>(endReadParallel - startReadParallel).count();
            std::cout << "[Parallel Read] Size=" << n << ", time=" << parallelReadTime << "s\n";

            auto startReadSequential = std::chrono::high_resolution_clock::now();
            readMatrixOneDim(fa, A, nA);
            readMatrixOneDim(fb, B, nB);
            auto endReadSequential = std::chrono::high_resolution_clock::now();
            double sequentialReadTime = std::chrono::duration<double>(endReadSequential - startReadSequential).count();
            std::cout << "[Sequential Read] Size=" << n << ", time=" << sequentialReadTime << "s\n";

        }


        for (int i = 0; i < n * n; i++) C[i] = 0.0;
        auto startSeq = std::chrono::high_resolution_clock::now();
        multiplySequential(A, B, C, n);
        auto endSeq = std::chrono::high_resolution_clock::now();
        {
            std::string fileC = "C_" + std::to_string(n) + "_seq.txt";
            writeMatrixOneDim(fileC, C, n);
        }
        seqTimes[si] = std::chrono::duration<double>(endSeq - startSeq).count();
        std::cout << "[SEQ] n=" << n << " time=" << seqTimes[si] << "s\n";

        for (size_t tj = 0; tj < parThreads.size(); tj++){
            int t = parThreads[tj];

            for (int i = 0; i < n * n; i++) C[i] = 0.0;
            auto startI = std::chrono::high_resolution_clock::now();
            multiplyParallelI(A, B, C, n, t);
            auto endI = std::chrono::high_resolution_clock::now();
            {
                std::string fileC = "C_" + std::to_string(n) + "_T" + std::to_string(t) + "_i.txt";
                writeMatrixOneDim(fileC, C, n);
            }
            iTimes[si][tj] = std::chrono::duration<double>(endI - startI).count();
            std::cout << "[PAR_I] n=" << n << " threads=" << t << " time=" << iTimes[si][tj] << "s\n";

            for (int i = 0; i < n * n; i++) C[i] = 0.0;
            auto startJ = std::chrono::high_resolution_clock::now();
            multiplyParallelJ(A, B, C, n, t);
            auto endJ = std::chrono::high_resolution_clock::now();
            {
                std::string fileC = "C_" + std::to_string(n) + "_T" + std::to_string(t) + "_j.txt";
                writeMatrixOneDim(fileC, C, n);
            }
            jTimes[si][tj] = std::chrono::duration<double>(endJ - startJ).count();
            std::cout << "[PAR_J] n=" << n << " threads=" << t << " time=" << jTimes[si][tj] << "s\n";

            for (int i = 0; i < n * n; i++) C[i] = 0.0;
            auto startK = std::chrono::high_resolution_clock::now();
            multiplyParallelK(A, B, C, n, t);
            auto endK = std::chrono::high_resolution_clock::now();
            {
                std::string fileC = "C_" + std::to_string(n) + "_T" + std::to_string(t) + "_k.txt";
                writeMatrixOneDim(fileC, C, n);
            }
            kTimes[si][tj] = std::chrono::duration<double>(endK - startK).count();
            std::cout << "[PAR_K] n=" << n << " threads=" << t << " time=" << kTimes[si][tj] << "s\n";

            std::cout << "-------------------------------------------\n";
        }

        delete[] A;
        delete[] B;
        delete[] C;
        std::cout << "==============================================\n";
    }

    {
        std::ofstream foutSeq("SeqTable.txt");
        foutSeq << "N\tTime(s)\n";
        for (size_t si = 0; si < sizes.size(); si++){
            foutSeq << sizes[si] << "\t" << seqTimes[si] << "\n";
        }
        foutSeq.close();
    }
    {
        std::ofstream foutParI("ParI_Table.txt");
        foutParI << "N\tThreads\tTime(s)\n";
        for (size_t si = 0; si < sizes.size(); si++){
            for (size_t tj = 0; tj < parThreads.size(); tj++){
                foutParI << sizes[si] << "\t" << parThreads[tj] << "\t" << iTimes[si][tj] << "\n";
            }
        }
        foutParI.close();
    }
    {
        std::ofstream foutParJ("ParJ_Table.txt");
        foutParJ << "N\tThreads\tTime(s)\n";
        for (size_t si = 0; si < sizes.size(); si++){
            for (size_t tj = 0; tj < parThreads.size(); tj++){
                foutParJ << sizes[si] << "\t" << parThreads[tj] << "\t" << jTimes[si][tj] << "\n";
            }
        }
        foutParJ.close();
    }
    {
        std::ofstream foutParK("ParK_Table.txt");
        foutParK << "N\tThreads\tTime(s)\n";
        for (size_t si = 0; si < sizes.size(); si++){
            for (size_t tj = 0; tj < parThreads.size(); tj++){
                foutParK << sizes[si] << "\t" << parThreads[tj] << "\t" << kTimes[si][tj] << "\n";
            }
        }
        foutParK.close();
    }
    std::vector<int> sizesSched = {1000, 2000};
    std::vector<int> threadsSched = {8, 16};
    std::vector<int> chunkSizes = {10, 50, 100};
    std::vector<SchedResult> schedResults;
    for (int n : sizesSched) {
        double *A = new double[n * n];
        double *B = new double[n * n];
        double *C = new double[n * n];
        {
            std::string fa = "A_" + std::to_string(n) + ".txt";
            std::string fb = "B_" + std::to_string(n) + ".txt";
            int nA = 0, nB = 0;
            auto sRead = std::chrono::high_resolution_clock::now();
#pragma omp parallel sections
            {
#pragma omp section
                {
                    readMatrixOneDim(fa, A, nA);
                }
#pragma omp section
                {
                    readMatrixOneDim(fb, B, nB);
                }
            }
            auto eRead = std::chrono::high_resolution_clock::now();
            double tRead = std::chrono::duration<double>(eRead - sRead).count();
            std::cout << "Schedule Test: Size=" << n << ", file read time=" << tRead << "s\n";
        }
        for (int t : threadsSched) {
            for (size_t s = 0; s < 3; s++) {
                for (int chunk : chunkSizes) {
                    auto startSched = std::chrono::high_resolution_clock::now();
                    if (s == 0) {
#pragma omp parallel for num_threads(t) schedule(static, chunk)
                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < n; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < n; k++) {
                                    sum += A[i * n + k] * B[k * n + j];
                                }
                                C[i * n + j] = sum;
                            }
                        }
                    } else if (s == 1) {
#pragma omp parallel for num_threads(t) schedule(dynamic, chunk)
                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < n; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < n; k++) {
                                    sum += A[i * n + k] * B[k * n + j];
                                }
                                C[i * n + j] = sum;
                            }
                        }
                    } else {
#pragma omp parallel for num_threads(t) schedule(guided, chunk)
                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < n; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < n; k++) {
                                    sum += A[i * n + k] * B[k * n + j];
                                }
                                C[i * n + j] = sum;
                            }
                        }
                    }
                    auto endSched = std::chrono::high_resolution_clock::now();
                    double timeSched = std::chrono::duration<double>(endSched - startSched).count();

                    std::string fileC = "C_" + std::to_string(n) + "_T" + std::to_string(t) +
                                        "_sched_" + (s == 0 ? "static" : (s == 1 ? "dynamic" : "guided")) +
                                        "_" + std::to_string(chunk) + ".txt";
                    writeMatrixOneDim(fileC, C, n);

                    SchedResult res = {n, t, (s == 0 ? "static" : (s == 1 ? "dynamic" : "guided")), chunk, timeSched};
                    schedResults.push_back(res);
                    std::cout << "[PAR_I_Sched] n=" << n << " threads=" << t
                              << " schedule=" << (s == 0 ? "static" : (s == 1 ? "dynamic" : "guided"))
                              << " chunk=" << chunk << " time=" << timeSched << "s\n";
                }
            }
        }

        delete[] A;
        delete[] B;
        delete[] C;
        std::cout << "----------------------------------------------\n";
    }
    {
        std::ofstream foutSched("ParI_Sched_Table.txt");
        foutSched << "N\tThreads\tSchedule\tChunk\tTime(s)\n";
        for (const auto &r : schedResults) {
            foutSched << r.n << "\t" << r.threads << "\t" << r.sched
                      << "\t" << r.chunk << "\t" << r.time << "\n";
        }
        foutSched.close();
    }
    std::cout << "\nTables have been saved to files: SeqTable.txt, ParI_Table.txt, ParJ_Table.txt, ParK_Table.txt, ParI_Sched_Table.txt\n";
    return 0;
}

