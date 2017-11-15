//
//  main.cpp
//  matrix-multiply-par
//
//  Created by Adam Vongrej on 11/14/17.
//  Copyright © 2017 Adam Vongrej. All rights reserved.
//

#include <iostream>
#include <omp.h>

extern "C" {
    #include <immintrin.h>
}

const int BLOCK_SIZE = 128;

//typedef unsigned short val;
typedef double val;

int seed = 0;

long int hash(long int a, long int b) {
    return (a | a << 27) * (b + 2352351);
}

int next_int() {
    seed = (seed + 234532) * ((seed >> 5) + 12234);
    return seed & 16383;
}

void multiply(val** a, val** b, val** c, int size) {
    
    int x,y,i;
    
    #pragma omp parallel for private(x,y,i)
    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {
            int sum = 0;
            for (i = 0; i < size; i++) {
                sum += a[x][i] * b[y][i];
            }
            c[x][y] = sum;
        }
    }
}

void multiply_avx(val** a, val** b, val** c, int size) {

    // Each vector can have 4 double precisions floating points
    int avx_vector_size = 4;
    int diff = size % avx_vector_size;
    int avx_portion = size - diff;

    int x,y,i;

    #pragma omp parallel for private(x,y,i)
    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {

            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

            val sum = 0;
            val out[avx_vector_size];

            // Luckily m == n
            for (i = 0; i < avx_portion; i += 8 * avx_vector_size) {
                // Load vectors
                ymm0 = _mm256_loadu_pd(&a[x][i]);
                ymm1 = _mm256_loadu_pd(&a[x][i + 4]);
                ymm2 = _mm256_loadu_pd(&a[x][i + 8]);
                ymm3 = _mm256_loadu_pd(&a[x][i + 12]);
                ymm4 = _mm256_loadu_pd(&a[x][i + 16]);
                ymm5 = _mm256_loadu_pd(&a[x][i + 20]);
                ymm6 = _mm256_loadu_pd(&a[x][i + 24]);
                ymm7 = _mm256_loadu_pd(&a[x][i + 28]);

                ymm8 = _mm256_loadu_pd(&b[y][i]);
                ymm9 = _mm256_loadu_pd(&b[y][i + 4]);
                ymm10 = _mm256_loadu_pd(&b[y][i + 8]);
                ymm11 = _mm256_loadu_pd(&b[y][i + 12]);
                ymm12 = _mm256_loadu_pd(&b[y][i + 16]);
                ymm13 = _mm256_loadu_pd(&b[y][i + 20]);
                ymm14 = _mm256_loadu_pd(&b[y][i + 24]);
                ymm15 = _mm256_loadu_pd(&b[y][i + 28]);

                // Multiply vectors
                ymm0 = _mm256_mul_pd(ymm0, ymm8);
                ymm1 = _mm256_mul_pd(ymm1, ymm9);
                ymm2 = _mm256_mul_pd(ymm2, ymm10);
                ymm3 = _mm256_mul_pd(ymm3, ymm11);
                ymm4 = _mm256_mul_pd(ymm4, ymm12);
                ymm5 = _mm256_mul_pd(ymm5, ymm13);
                ymm6 = _mm256_mul_pd(ymm6, ymm14);
                ymm7 = _mm256_mul_pd(ymm7, ymm15);

                // Add vectors
                ymm0 = _mm256_add_pd(ymm0, ymm1);
                ymm2 = _mm256_add_pd(ymm2, ymm3);
                ymm4 = _mm256_add_pd(ymm4, ymm5);
                ymm6 = _mm256_add_pd(ymm6, ymm7);

                ymm0 = _mm256_add_pd(ymm0, ymm2);
                ymm4 = _mm256_add_pd(ymm4, ymm6);

                ymm0 = _mm256_add_pd(ymm0, ymm4);

                // Store vector
                // An unaligned store is required because the arrays are not guaranteed to be properly aligned in memory for AVX.
                _mm256_storeu_pd(out, ymm0);

                for (int i = 0; i < avx_vector_size; i++) {
                    sum += out[i];
                }
            }

            // Calculate remainder the standard way
            for (i = avx_portion; i < size; i++) {
                sum += a[x][i] * b[y][i];
            }

            c[x][y] = sum;
        }
    }
}

// Perfomance decrease!!
void multiply_avx2(val** a, val** b, val** c, int size) {

    // Each vector can have 4 double precisions floating points
    int avx_vector_size = 4;
    int diff = size % avx_vector_size;
    int avx_portion = size - diff;

    int x,y,i;

    // Since we have 4 threads in total, reduce the number of ymm registers to 16/4 = 4 per iteration (thread)
    #pragma omp parallel for collapse(2) private(x,y,i) num_threads(omp_get_num_procs())
    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {

            __m256d ymm0, ymm1, ymm2, ymm3;

            val sum = 0;
            val out[avx_vector_size];

            // Luckily m == n
            for (i = 0; i < avx_portion; i += 2 * avx_vector_size) {
                // Load vectors
                ymm0 = _mm256_loadu_pd(&a[x][i]);
                ymm1 = _mm256_loadu_pd(&a[x][i + 4]);

                ymm2 = _mm256_loadu_pd(&b[y][i]);
                ymm3 = _mm256_loadu_pd(&b[y][i + 4]);


                // Multiply vectors
                ymm0 = _mm256_mul_pd(ymm0, ymm2);
                ymm1 = _mm256_mul_pd(ymm1, ymm3);

                // Add vectors
                ymm0 = _mm256_add_pd(ymm0, ymm1);

                // Store vector
                // An unaligned store is required because the arrays are not guaranteed to be properly aligned in memory for AVX.
                _mm256_storeu_pd(out, ymm0);

                for (int i = 0; i < avx_vector_size; i++) {
                    sum += out[i];
                }
            }

            // Calculate remainder the standard way
            for (i = avx_portion; i < size; i++) {
                sum += a[x][i] * b[y][i];
            }

            c[x][y] = sum;
        }
    }
}

void multiply_tiled(val** a, val** b, val** c, int size) {
    int x,y,k, xx, yy, kk;

    #pragma omp parallel for collapse(2) private(x,y,k,xx,yy,kk) shared(a,b,c)
    for (xx = 0; xx < size; xx += BLOCK_SIZE) {
        for (yy = 0; yy < size; yy += BLOCK_SIZE) {
            for (kk = 0; kk < size; kk+= BLOCK_SIZE) {
                // b * b mini matrix multiplication
                for (x = xx; x < std::min(xx + BLOCK_SIZE, size); x++) {
                    for (y = yy; y < std::min(yy + BLOCK_SIZE, size); y++) {
                        int sum = 0;
                        for (k = kk; k < std::min(kk + BLOCK_SIZE, size); k++) {
                            c[x][y] += a[x][k] * b[y][k];
                        }
                        c[x][y] += sum;
                    }
                }
            }
        }
    }
}

void multiply_tiled2(val** a, val** b, val** c, int size) {

    for(int xx = 0; xx < size; xx += BLOCK_SIZE) {
        for(int yy = 0; yy < size; yy += BLOCK_SIZE) {
            for(int kk = 0; kk < size; kk += BLOCK_SIZE) {
                #pragma omp parallel for num_threads(omp_get_num_procs())
                for(int x = xx; x < std::min(xx + BLOCK_SIZE, size); x++) {
                    for(int k = kk; k < std::min(kk + BLOCK_SIZE, size); k++) {
                        for(int y = yy; y < std::min(yy + BLOCK_SIZE, size); y++) {
                            c[x][y] += a[x][k] * b[y][k];
                        }
                    }
                }
            }
        }
    }
}

void multiply_tiled_x(val** a, val** b, val** c, int size) {
    int x,y,k, xx, kk;
    
    #pragma omp parallel for private(x,y,k,xx,kk) shared(a,b,c)
    for (xx = 0; xx < size; xx += BLOCK_SIZE) {
        for (kk = 0; kk < size; kk += BLOCK_SIZE) {
            for (y = 0; y < size; y++) {
                for (x = xx; x < std::min(xx + BLOCK_SIZE, size); x++) {
                    for (k = kk; k < std::min(kk + BLOCK_SIZE, size); k++) {
                        c[x][y] += a[x][k] * b[y][k];
                    }
                }
            }
        }
    }
}

void multiply_cubed(val** a, val** b, val** c, int size) {
    int x,y,k, xx, yy, kk;
    
    val*** cc = new val**[size];
    for (int x = 0; x < size; x++) {
        cc[x] = new val*[size];
        for (int y = 0; y < size; y++) {
            cc[x][y] = new val[size];
        }
    }
    
    
    //#pragma omp parallel for schedule(static, 1) collapse(5) private(x,y,k,xx,yy,kk) shared(a,b,cc)
    #pragma omp parallel for collapse(3) private(x,y,k,xx,yy,kk)
    for (xx = 0; xx < size; xx += BLOCK_SIZE) {
        for (yy = 0; yy < size; yy += BLOCK_SIZE) {
            for (kk = 0; kk < size; kk+= BLOCK_SIZE) {
                // b * b mini matrix multiplication
                for (x = xx; x < std::min(xx + BLOCK_SIZE, size); x++) {
                    for (y = yy; y < std::min(yy + BLOCK_SIZE, size); y++) {
                        for (k = kk; k < std::min(kk + BLOCK_SIZE, size); k++) {
                            cc[x][y][k] = a[x][k] * b[y][k];
                        }
                    }
                }
            }
        }
    }
    
    int z = 0;
    
    #pragma omp parallel for collapse(2) private(x,y,z) shared(c,cc)
    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {
            for (z = 0; z < size; z++) {
                c[x][y] += cc[x][y][z];
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    
    int size = atoi(argv[1]);
    
    seed = atoi(argv[2]);
    
    //omp_set_num_threads(4);
    
    val** a = new val*[size];
    val** b = new val*[size];
    val** c = new val*[size];
    
    for (int x = 0; x < size; x++) {
        a[x] = new val[size];
        b[x] = new val[size];
        c[x] = new val[size];
    }
    
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            a[x][y] = next_int();
            b[y][x] = next_int();
            c[x][y] = 0;
        }
    }
    
    //multiply(a, b, c, size);
    //multiply_tiled(a, b, c, size);
    //multiply_tiled2(a, b, c, size);
    //multiply_tiled_x(a, b, c, size);
    //multiply_cubed(a, b, c, size);
    multiply_avx(a, b, c, size);
    //multiply_avx2(a, b, c, size);
    
    int h = atoi(argv[2]);
    
    for (int k = 0; k < 3; k++) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                h = hash(h, (long int)c[x][y]);
            }
        }
    }
    
    printf("%d\n", h & 1023);
    
    return 0;
}


