/*
 * Copyright (c) 2021, Dirk Pleiter, KTH
 *
 * This source code is in parts based on code from Jiri Kraus (NVIDIA) and
 * Andreas Herten (Forschungszentrum Juelich)
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND Any
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR Any DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON Any THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN Any WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SUB_ITER 100

int
solver(double *v, double *f, int nx, int ny, double eps, int nmax) {
  int n_cores = 256;
  int n_threads;

  const int size = nx * ny;
  int       n    = 0;
  double    e    = 2. * eps;
  double   *vp;

  // temp approximate solution vector
  vp = (double *)malloc(nx * ny * sizeof(double));

  double w = 0.0;

  // loop until convergence
#pragma omp parallel num_threads(n_cores)
  {
    n_threads = omp_get_num_threads();

      while ((e > eps) && (n < nmax)) {
#pragma omp barrier
          for (int i = 0; i < SUB_ITER / 2; i++) {
#pragma omp for
              for (int iy = 1; iy < (ny - 1); iy++) {
                  for (int ix = 1; ix < (nx - 1); ix++) {
                    int lin_idx = nx * iy + ix;
                    // compute v^{k+1}
                    vp[lin_idx]                  // v_{i,j}
                      = -0.25                    // 1/t_{i,i} --> this is D^-1
                      * (f[lin_idx]              // f_{i,j}
                         - (v[lin_idx + 1]       // v_{i+1,j} -->
                            + v[lin_idx - 1]     // v_{i-1,j} --> this is R*v
                            + v[lin_idx + nx]    // v_{i,j+1} -->
                            + v[lin_idx - nx])); // v_{i,j-1} -->
                  }
              }
              // updated solution is in vp
              // apply boundary on vp
#pragma omp for
              for (int ix = 1; ix < (nx - 1); ix++) {
                // y = 0
                vp[nx * 0 + ix] = vp[nx * (ny - 2) + ix];
                // y = NY
                vp[nx * (ny - 1) + ix] = vp[nx * 1 + ix];
              }
#pragma omp for
              for (int iy = 1; iy < (ny - 1); iy++) {
                // x = 0
                vp[nx * iy + 0] = vp[nx * iy + (nx - 2)];
                // x = NX
                vp[nx * iy + (nx - 1)] = vp[nx * iy + 1];
              }
              // do the same but switch arrays v and vp

#pragma omp for
              for (int iy = 1; iy < (ny - 1); iy++) {
                  for (int ix = 1; ix < (nx - 1); ix++) {
                    int lin_idx = nx * iy + ix;
                    // compute v^{k+1}
                    v[lin_idx]                    // v_{i,j}
                      = -0.25                     // 1/t_{i,i} --> this is D^-1
                      * (f[lin_idx]               // f_{i,j}
                         - (vp[lin_idx + 1]       // v_{i+1,j} -->
                            + vp[lin_idx - 1]     // v_{i-1,j} --> this is R*v
                            + vp[lin_idx + nx]    // v_{i,j+1} -->
                            + vp[lin_idx - nx])); // v_{i,j-1} -->
                  }
              }
              // updated solution is in v
              // apply boundary on v
#pragma omp for
              for (int ix = 1; ix < (nx - 1); ix++) {
                // y = 0
                v[nx * 0 + ix] = v[nx * (ny - 2) + ix];
                // y = NY
                v[nx * (ny - 1) + ix] = v[nx * 1 + ix];
              }
#pragma omp for
              for (int iy = 1; iy < (ny - 1); iy++) {
                // x = 0
                v[nx * iy + 0] = v[nx * iy + (nx - 2)];
                // x = NX
                v[nx * iy + (nx - 1)] = v[nx * iy + 1];
              }
          }
          // compute the error after SUB_ITER

#pragma omp master
        {
          e = 0.0;
          w = 0.0;
        }

#pragma omp for reduction(+ : w) reduction(max : e)
          for (int iy = 0; iy < ny; iy++) {
              for (int ix = 0; ix < nx; ix++) {
                int    lin_idx = nx * iy + ix;
                double d       = fabs(v[lin_idx] - vp[lin_idx]);
                e              = (d > e) ? d : e;
                w += fabs(v[lin_idx]);
              }
          }

#pragma omp master
        {
          w /= size;
          e /= w;
          n += SUB_ITER;
        }
#pragma omp barrier
      }
  }
  printf("FINISHED!\n");

  free(vp);

    if (e < eps) {
      printf("n_cores: %d - n_threads: %d\n", n_cores, n_threads);
      printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e);
  } else
    printf("ERROR: Failed to converge (%d)\n", n);

  return (e < eps ? 0 : 1);
}
