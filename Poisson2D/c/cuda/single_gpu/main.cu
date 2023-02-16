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
#include <time.h>

#ifndef NX
#  define NX 128
#endif
#ifndef NY
#  define NY 128
#endif
#ifndef BLOCK_SIZE
#  define BLOCK_SIZE 16
#endif
#define NMAX 100
#define SUB_ITER 100
#define EPS 1e-5

__global__ void
solver(double * /*in*/, double * /*forcing term*/, int /*NX*/, int /*NY*/, double * /*out*/);

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
  double *v;
  double *vp;
  double *f;
  double *w;
  double *e;

  int size = NX * NY;

  dim3 jacobiGrid((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 jacobiBlock(BLOCK_SIZE, BLOCK_SIZE);

  int block_dim = BLOCK_SIZE * BLOCK_SIZE;

  dim3 errorGrid(((NX * NY) + block_dim - 1) / block_dim);
  dim3 errorBlock(block_dim);

  // Allocate memory
  // approximate solution vector & update vector
  cudaMallocManaged(&v, NX * NY * sizeof(double));
  cudaMallocManaged(&vp, NX * NY * sizeof(double));
  // forcing term
  cudaMallocManaged(&f, NX * NY * sizeof(double));
  // error
  cudaMallocManaged(&w, errorGrid.x * sizeof(double));
  cudaMallocManaged(&e, errorGrid.x * sizeof(double));

  printf("Matrix size: %d\n", size);
  printf("NX=%d;NY=%d;\n", NX, NY);

  // Initialize input
    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
          // initial guess is 0
          v[NX * iy + ix]  = 0.0;
          vp[NX * iy + ix] = 0.0;

          const double x = 2.0 * ix / (NX - 1.0) - 1.0;
          const double y = 2.0 * iy / (NY - 1.0) - 1.0;
          // forcing term is a sinusoid
          f[NX * iy + ix] = sin(x + y);
        }
    }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  double w_sum;
  double e_max = 2 * EPS;

  int n = 0;
  cudaEventRecord(start, 0);
    while (e_max > EPS) {
        for (int i = 0; i < SUB_ITER / 2; i++) {
          solver<<<jacobiGrid, jacobiBlock>>>(v, f, NX, NY, vp);
          apply_bcs<<<jacobiGrid, jacobiBlock>>>(vp, NX, NY);
          // vp - updated solution
          solver<<<jacobiGrid, jacobiBlock>>>(vp, f, NX, NY, v);
          apply_bcs<<<jacobiGrid, jacobiBlock>>>(v, NX, NY);
          // v - updated solution
        }

      compute_error<<<errorGrid, errorBlock>>>(v, vp, size, e, w);
      cudaDeviceSynchronize();
      e_max = 0;
      w_sum = 0;
        for (int i = 0; i < errorGrid.x; i++) {
          e_max = (e[i] > e_max) ? e[i] : e_max;
          w_sum += w[i];
        }

      w_sum /= size;
      e_max /= w_sum;

      n++;
    }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  float dt;
  cudaEventElapsedTime(&dt, start, end);
  printf(
    "Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n * SUB_ITER, NX, NY, e_max);
  printf("Time elapsed: %f[ms]\n", dt);

  const char *filename = "solution.csv";

  FILE *file = fopen(filename, "w");
  fprintf(file, "x,y,v\n");
  for (int iy = 0; iy < NY; iy++)
    for (int ix = 0; ix < NX; ix++)
      fprintf(file, "%d,%d,%lf\n", ix, iy, v[iy * NX + ix]);
  fclose(file);

  printf("Output written to %s\n", filename);

  // Clean-up
  cudaFree(v);
  cudaFree(vp);
  cudaFree(f);

  return 0;
}