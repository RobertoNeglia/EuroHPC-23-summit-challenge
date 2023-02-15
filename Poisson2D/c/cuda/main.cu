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
#  define BLOCK_SIZE 8
#endif
#define NMAX 20000

__global__ void
solver(double * /*in*/, double * /*forcing term*/, int /*NX*/, int /*NY*/, double * /*out*/);

__global__ void
apply_boundary(double *, int, int);

int
main() {
  double *v;
  double *vp;
  double *f;
  double *sol;

  // Allocate memory
  // approximate solution vector & update vector
  cudaMallocManaged(&v, NX * NY * sizeof(double));
  cudaMallocManaged(&vp, NX * NY * sizeof(double));
  // forcing term
  cudaMallocManaged(&f, NX * NY * sizeof(double));

  printf("Matrix size: %d\n", NX * NY);

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

  printf("Data initialized\n");

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  dim3 dimGrid((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  int n = 0;
  cudaEventRecord(start, 0);
    while ((n < NMAX)) {
        // Call solver
        if (n % 2) {
          solver<<<dimGrid, dimBlock>>>(vp, f, NX, NY, v);
          cudaDeviceSynchronize();
          apply_boundary<<<dimGrid, dimBlock>>>(v, NX, NY);
        } else {
          solver<<<dimGrid, dimBlock>>>(v, f, NX, NY, vp);
          cudaDeviceSynchronize();
          apply_boundary<<<dimGrid, dimBlock>>>(vp, NX, NY);
        }
      cudaDeviceSynchronize();
      n++;
    }
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

    if (n % 2) {
      sol = v;
    } else {
      sol = vp;
    }

  float dt;
  cudaEventElapsedTime(&dt, start, end);
  printf("Time elapsed: %f[ms]\n", dt);

  const char *filename = "solution.csv";

  FILE *file = fopen(filename, "w");
  fprintf(file, "x,y,v\n");
  for (int iy = 0; iy < NY; iy++)
    for (int ix = 0; ix < NX; ix++)
      fprintf(file, "%d,%d,%lf\n", ix, iy, sol[iy * NX + ix]);
  fclose(file);

  printf("Output written to %s\n", filename);

  // Clean-up
  cudaFree(v);
  cudaFree(vp);
  cudaFree(f);

  return 0;
}