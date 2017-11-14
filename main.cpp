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
    
    //#pragma omp parallel for schedule(static,50) collapse(2) private(x,y,i) shared(a,b,c)
    //#pragma omp parallel for schedule(dynamic, 10) collapse(2) private(x,y,i) shared(a,b,c)
    
    
    int x,y,i;
    
    #pragma omp parallel for private(x,y,i)
    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {
            double sum = 0;
            for (i = 0; i < size; i++) {
                sum += a[x][i] * b[y][i];
            }
            c[x][y] = sum;
        }
    }
}

void multiply_avx(val** a, val** b, val** c, int size) {

    int x,y,i;

    int avx_block = 8;
    int diff = size % avx_block;
    int avx_size = size - diff;

    for (x = 0; x < size; x++) {
        for (y = 0; y < size; y++) {

            __m256 a1_avx, a2_avx, a3_avx, a4_avx, a5_avx, a6_avx, a7_avx, a8_avx, b1_avx, b2_avx, b3_avx, b4_avx, b5_avx, b6_avx, b7_avx, b8_avx;
            float sum_arr[8];
            double sum = 0;

            // Luckily m == n
            for (i = 0; i < avx_size; i += 8 * avx_block) {
                a1_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i]));
                a2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 8]));
                a3_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 16]));
                a4_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 24]));
                a5_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 32]));
                a6_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 40]));
                a7_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 48]));
                a8_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&a[x][i + 56]));

                b1_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 8]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 16]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 24]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 32]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 40]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 48]));
                b2_avx = __builtin_ia32_loadups256(reinterpret_cast<float*>(&b[y][i + 56]));

                a1_avx = __builtin_ia32_mulps256(a1_avx, b1_avx);
                a2_avx = __builtin_ia32_mulps256(a2_avx, b2_avx);
                a3_avx = __builtin_ia32_mulps256(a3_avx, b3_avx);
                a4_avx = __builtin_ia32_mulps256(a4_avx, b4_avx);
                a5_avx = __builtin_ia32_mulps256(a5_avx, b5_avx);
                a6_avx = __builtin_ia32_mulps256(a6_avx, b6_avx);
                a7_avx = __builtin_ia32_mulps256(a7_avx, b7_avx);
                a8_avx = __builtin_ia32_mulps256(a8_avx, b8_avx);

                a1_avx = __builtin_ia32_addps256(a1_avx, a2_avx);
                a2_avx = __builtin_ia32_addps256(a3_avx, a4_avx);
                a3_avx = __builtin_ia32_addps256(a5_avx, a6_avx);
                a4_avx = __builtin_ia32_addps256(a7_avx, a8_avx);

                a1_avx = __builtin_ia32_addps256(a1_avx, a2_avx);
                a2_avx = __builtin_ia32_addps256(a3_avx, a4_avx);

                a1_avx = __builtin_ia32_addps256(a1_avx, a2_avx);

                __builtin_ia32_storeups256(sum_arr, a1_avx);

                for (int i = 0; i < avx_size; i++) {
                    sum += sum_arr[i];
                }
            }

            // Calculate remainder the standard way
            for (i = avx_size; i < avx_size + diff; i++) {
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


