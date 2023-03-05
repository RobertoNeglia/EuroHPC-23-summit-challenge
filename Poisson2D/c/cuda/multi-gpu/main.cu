/*
 * Copyright (c) 2021, Dirk Pleiter, KTH
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NX
#  define NX 32
#endif
#ifndef NY
#  define NY 32
#endif
#ifndef BLOCK_SIZE
#  define BLOCK_SIZE 8
#endif
#define NMAX 2
#define SUB_ITER 2
#define EPS 1e-5

__global__ void
solver(const double * /*in*/,
       const double * /*forcing term*/,
       const int /*NX*/,
       const int /*NY*/,
       double * /*out*/);

__global__ void
apply_bcs(double * /*sol*/, const int /*NX*/, const int /*NY*/);

__global__ void
compute_error(const double * /*sol*/,
              const double * /*old_sol*/,
              const int /*size (NX*NY)*/,
              double * /*error*/,
              double * /*weight*/);

int
main() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Number of GPUs: %d\n", deviceCount);

  cudaStream_t streams[deviceCount];

  int    sol_size     = NX * NY;
  int    loc_sol_size = sol_size / deviceCount; // assume sol_size is divisible by deviceCount
  int    loc_ny       = NY / deviceCount;
  size_t sol_local_bytes = loc_sol_size * sizeof(double);

  printf("Matrix size: %d\n", sol_size);
  printf("Local solution size: %d\n", loc_sol_size);
  printf("NX=%d;NY=%d;\n", NX, NY);

  int block_dim = BLOCK_SIZE * BLOCK_SIZE;

  dim3 jacobiBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 jacobiGrid(((NX - 2) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  ((loc_ny - 2) + BLOCK_SIZE - 1) / BLOCK_SIZE);

  dim3 errorBlock(block_dim);
  dim3 errorGrid(((loc_sol_size) + block_dim - 1) / block_dim);

  int    diff_size  = errorGrid.x;
  size_t diff_bytes = diff_size * sizeof(double);

  printf("Allocating memory...\n");
  // host and device arrays
  double *h_v[deviceCount], *d_v[deviceCount];
  double *h_vp[deviceCount], *d_vp[deviceCount];
  double *h_f[deviceCount], *d_f[deviceCount];
  double *h_w[deviceCount], *d_w[deviceCount];
  double *h_e[deviceCount], *d_e[deviceCount];

    for (int dev = 0; dev < deviceCount; dev++) { // Host memory allocation
      // approximate solution vector & update vector
      h_v[dev]  = (double *)calloc(loc_sol_size, sizeof(double));
      h_vp[dev] = (double *)calloc(loc_sol_size, sizeof(double));
      // forcing term
      h_f[dev] = (double *)calloc(loc_sol_size, sizeof(double));
      // weight
      h_w[dev] = (double *)calloc(diff_size, sizeof(double));
      // error
      h_e[dev] = (double *)calloc(diff_size, sizeof(double));

      // Device memory allocation
      cudaSetDevice(dev);
      cudaStreamCreate(&streams[dev]);
      // approximate solution vector & update vector
      cudaMalloc(d_v + dev, sol_local_bytes);
      cudaMalloc(d_vp + dev, sol_local_bytes);
      // forcing term
      cudaMalloc(d_f + dev, sol_local_bytes);
      // weight
      cudaMalloc(d_w + dev, diff_bytes);
      // error
      cudaMalloc(d_e + dev, diff_bytes);
    }

  printf("Initializing input...\n");
    // Initialize input
    for (int dev = 0; dev < deviceCount; dev++) {
        for (int iy = 0; iy < loc_ny; iy++) {
            for (int ix = 0; ix < NX; ix++) {
              const double x = 2.0 * ix / (NX - 1.0) - 1.0;
              const double y = 2.0 * (iy + loc_ny * dev) / (NY - 1.0) - 1.0;
              // forcing term is a sinusoid
              h_f[dev][NX * iy + ix] = sin(x + y);
            }
        }
    }

    // initialization of forcing term ok

    // Copy data from host to devices
    for (int dev = 0; dev < deviceCount; dev++) {
      cudaSetDevice(dev);
      cudaMemcpy(d_v[dev], h_v[dev], loc_sol_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_vp[dev], h_vp[dev], loc_sol_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_f[dev], h_f[dev], loc_sol_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_w[dev], h_w[dev], diff_bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(d_e[dev], h_e[dev], diff_bytes, cudaMemcpyHostToDevice);
    }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  double w_sum;
  double e_max = 2 * EPS;

  int n = 0;
  cudaEventRecord(start, 0);
    while (e_max > EPS && n < NMAX) {
        for (int i = 0; i < SUB_ITER / 2; i++) {
            for (int dev = 0; dev < deviceCount; dev++) {
              cudaSetDevice(dev);
              solver<<<jacobiGrid, jacobiBlock /*, 0, streams[dev]*/>>>(
                d_v[dev], d_f[dev], NX, loc_ny, d_vp[dev]);
              cudaDeviceSynchronize();
            }
            for (int dev = 0; dev < deviceCount; dev++) {
              cudaSetDevice(dev);
              apply_bcs<<<jacobiGrid, jacobiBlock /*, 0, streams[dev]*/>>>(d_vp[dev],
                                                                           NX,
                                                                           loc_ny);
              cudaDeviceSynchronize();
            }
          // vp - updated solution

          // synchronization between gpus (maybe) needed

            for (int dev = 0; dev < deviceCount; dev++) {
              cudaSetDevice(dev);
              solver<<<jacobiGrid, jacobiBlock /*, 0, streams[dev]*/>>>(
                d_vp[dev], d_f[dev], NX, loc_ny, d_v[dev]);
              cudaDeviceSynchronize();
            }
            for (int dev = 0; dev < deviceCount; dev++) {
              cudaSetDevice(dev);
              apply_bcs<<<jacobiGrid, jacobiBlock /*, 0, streams[dev]*/>>>(d_v[dev],
                                                                           NX,
                                                                           loc_ny);
              cudaDeviceSynchronize();
            }
          // v - updated solution
        }

      // copy solution back to host
        for (int dev = 0; dev < deviceCount; dev++) {
          cudaSetDevice(dev);
          cudaMemcpy(h_v[dev], d_v[dev], loc_sol_size, cudaMemcpyDeviceToHost);
        }

      // compute residual
        for (int dev = 0; dev < deviceCount; dev++) {
          cudaSetDevice(dev);
          compute_error<<<errorGrid, errorBlock /*, 0, streams[dev]*/>>>(
            d_v[dev], d_vp[dev], loc_sol_size, d_e[dev], d_w[dev]);
          cudaDeviceSynchronize();
        }

        for (int dev = 0; dev < deviceCount; dev++) {
          cudaSetDevice(dev);
          cudaMemcpy(h_w[dev], d_w[dev], diff_bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(h_e[dev], d_e[dev], diff_bytes, cudaMemcpyDeviceToHost);
        }

      // finish reduce on host
      e_max = 0;
      w_sum = 0;
        for (int dev = 0; dev < deviceCount; dev++) {
            for (int i = 0; i < diff_size; i++) {
              e_max = (h_e[dev][i] > e_max) ? h_e[dev][i] : e_max;
              w_sum += h_w[dev][i];
            }
        }

      w_sum /= sol_size;
      e_max /= w_sum;

      n += SUB_ITER;
    }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  float dt;
  cudaEventElapsedTime(&dt, start, end);
  printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, NX, NY, e_max);
  printf("Time elapsed: %f[ms]\n", dt);

    // copy solution back to host
    for (int dev = 0; dev < deviceCount; dev++) {
      cudaSetDevice(dev);
      cudaMemcpy(h_v[dev], d_v[dev], loc_sol_size, cudaMemcpyDeviceToHost);
      cudaStreamDestroy(streams[dev]);
    }

  // write solution to file
  const char *filename = "solution.csv";
  FILE       *file     = fopen(filename, "w");
  fprintf(file, "x,y,v\n");
  for (int dev = 0; dev < deviceCount; dev++)
    for (int iy = 0; iy < loc_ny; iy++)
      for (int ix = 0; ix < NX; ix++)
        fprintf(file, "%d,%d,%lf\n", ix, iy + dev * loc_ny, h_v[dev][iy * NX + ix]);
  fclose(file);

  printf("Output written to %s\n", filename);

    // Clean-up
    for (int dev = 0; dev < deviceCount; dev++) {
      cudaFree(d_v[dev]);
      cudaFree(d_vp[dev]);
      cudaFree(d_f[dev]);
      cudaFree(d_w[dev]);
      cudaFree(d_e[dev]);

      free(h_v[dev]);
      free(h_vp[dev]);
      free(h_f[dev]);
      free(h_w[dev]);
      free(h_e[dev]);
    }

  return 0;
}
