#include <iostream>
#include <fstream>
#include <omp.h>

int main() {
    int sizes[] = {20000, 40000};
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int max_threads = 40;
    
    
    std::ofstream outfile("vector-matrix_benchmark.txt");
    outfile << "# Размер Потоков Время(сек) Ускорение" << std::endl;
    
    for (int s = 0; s < 2; s++) {
        int m = sizes[s];
        int n = sizes[s];
        double base_time = 0.0;
        
        double* a = new double[m * n];
        double* b = new double[n];
        double* c = new double[m];

        omp_set_num_threads(max_threads);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            int rows_per_thread = m / nth;
            int start = tid * rows_per_thread;
            int end = (tid == nth - 1) ? m : start + rows_per_thread;
                
            for (int i = start; i < end; i++) {
                for (int j = 0; j < n; j++) {
                    a[i * n + j] = i + j;
                }
                c[i] = 0.0;
            }
        }
            
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            b[j] = j;
        }

        for (int t = 0; t < 8; t++) {
            int num_threads = threads[t];
            
            
            omp_set_num_threads(num_threads);
            

            
            double start_time = omp_get_wtime();
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                int rows_per_thread = m / nth;
                int start_row = tid * rows_per_thread;
                int end_row = (tid == nth - 1) ? m : start_row + rows_per_thread;
                
                for (int i = start_row; i < end_row; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < n; j++) {
                        sum += a[i * n + j] * b[j];
                    }
                    c[i] = sum;
                }
            }
            
            double end_time = omp_get_wtime();
            double time = end_time - start_time;
            
            if (num_threads == 1) {
                base_time = time;
            }
            
            double speedup = base_time / time;
            
            std::cout << num_threads << " " << time << " " << speedup << std::endl;
            outfile << m << " " << num_threads << " " << time << " " << speedup << std::endl;
            
            delete[] a;
            delete[] b;
            delete[] c;
        }
    }
    
    outfile.close();
    return 0;
}