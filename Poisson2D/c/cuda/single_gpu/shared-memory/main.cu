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
#  define NX 2048
#endif
#ifndef NY
#  define NY 2048
#endif
#ifndef BLOCK_SIZE
#  define BLOCK_SIZE 32
#endif
#define NMAX 200000
#define SUB_ITER 100
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
  // host and device arrays
  double *h_v, *d_v;
  double *h_vp, *d_vp;
  double *h_f, *d_f;
  double *h_w, *d_w;
  double *h_e, *d_e;

  dim3 jacobiGrid(((NX - 2) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  ((NY - 2) + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 jacobiBlock(BLOCK_SIZE, BLOCK_SIZE);

  int    sol_size       = NX * NY;
  size_t sol_size_bytes = sol_size * sizeof(double);

  int block_dim = BLOCK_SIZE * BLOCK_SIZE;

  dim3 boundaryGrid(((NX - 2) + (NY - 2) + block_dim - 1) / block_dim, 2);
  dim3 boundaryBlock(block_dim);

  dim3 errorGrid(((sol_size) + block_dim - 1) / block_dim);
  dim3 errorBlock(block_dim);

  int    diff_size       = errorGrid.x;
  size_t diff_size_bytes = diff_size * sizeof(double);

  // Host memory allocation
  // approximate solution vector & update vector
  h_v  = (double *)calloc(sol_size, sizeof(double));
  h_vp = (double *)calloc(sol_size, sizeof(double));
  // forcing term
  h_f = (double *)calloc(sol_size, sizeof(double));
  // weight
  h_w = (double *)calloc(diff_size, sizeof(double));
  // error
  h_e = (double *)calloc(diff_size, sizeof(double));

  // Device memory allocation
  // approximate solution vector & update vector
  cudaMalloc(&d_v, sol_size_bytes);
  cudaMalloc(&d_vp, sol_size_bytes);
  // forcing term
  cudaMalloc(&d_f, sol_size_bytes);
  // weight
  cudaMalloc(&d_w, diff_size_bytes);
  // error
  cudaMalloc(&d_e, diff_size_bytes);

  printf("Matrix sol_size: %d\n", sol_size);
  printf("NX=%d;NY=%d;\n", NX, NY);

  // Initialize input
    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
          const double x = 2.0 * ix / (NX - 1.0) - 1.0;
          const double y = 2.0 * iy / (NY - 1.0) - 1.0;
          // forcing term is a sinusoid
          h_f[NX * iy + ix] = sin(x + y);
        }
    }

  // Copy data from host to device
  cudaMemcpy(d_v, h_v, sol_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vp, h_vp, sol_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, h_f, sol_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w, diff_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_e, h_e, diff_size_bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  double w_sum;
  double e_max = 2 * EPS;

  int n = 0;
  cudaEventRecord(start, 0);
    while (e_max > EPS && n < NMAX) {
        for (int i = 0; i < SUB_ITER / 2; i++) {
          solver<<<jacobiGrid, jacobiBlock>>>(d_v, d_f, NX, NY, d_vp);
          apply_bcs<<<boundaryGrid, boundaryBlock>>>(d_vp, NX, NY);
          // vp - updated solution
          solver<<<jacobiGrid, jacobiBlock>>>(d_vp, d_f, NX, NY, d_v);
          apply_bcs<<<boundaryGrid, boundaryBlock>>>(d_v, NX, NY);
          // v - updated solution
        }
      compute_error<<<errorGrid, errorBlock>>>(d_v, d_vp, sol_size, d_e, d_w);
      cudaDeviceSynchronize();
      cudaMemcpy(h_w, d_w, diff_size_bytes, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_e, d_e, diff_size_bytes, cudaMemcpyDeviceToHost);
      // finish reduce on host
      e_max = 0;
      w_sum = 0;
        for (int i = 0; i < diff_size; i++) {
          e_max = (h_e[i] > e_max) ? h_e[i] : e_max;
          w_sum += h_w[i];
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
  cudaMemcpy(h_v, d_v, sol_size_bytes, cudaMemcpyDeviceToHost);

  const char *filename = "solution.csv";

  FILE *file = fopen(filename, "w");
  fprintf(file, "x,y,v\n");
  for (int iy = 0; iy < NY; iy++)
    for (int ix = 0; ix < NX; ix++)
      fprintf(file, "%d,%d,%lf\n", ix, iy, h_v[iy * NX + ix]);
  fclose(file);

  printf("Output written to %s\n", filename);

  // Clean-up
  cudaFree(d_v);
  cudaFree(d_vp);
  cudaFree(d_f);
  cudaFree(d_w);
  cudaFree(d_e);

  free(h_v);
  free(h_vp);
  free(h_f);
  free(h_w);
  free(h_e);

  return 0;
}
