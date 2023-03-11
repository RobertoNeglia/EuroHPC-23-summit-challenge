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
#include <stdio.h>
#include <stdlib.h>

#define SUB_ITER 100

void
solve(double *v, double *vp, double *f, int nx, int ny) {
  // loop over each element of the discretized domain
    for (int iy = 1; iy < (ny - 1); iy++) {
        for (int ix = 1; ix < (nx - 1); ix++) {
          int lin_idx = iy * nx + ix;
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
}

void
apply_boundary(double *v, int nx, int ny) {
  // compute weight on boundaries & apply boundary conditions
    for (int ix = 1; ix < (nx - 1); ix++) {
      // y = 0
      v[nx * 0 + ix] = v[nx * (ny - 2) + ix];
      // y = NY
      v[nx * (ny - 1) + ix] = v[nx * 1 + ix];
    }

    for (int iy = 1; iy < (ny - 1); iy++) {
      // x = 0
      v[nx * iy + 0] = v[nx * iy + (nx - 2)];
      // x = NX
      v[nx * iy + (nx - 1)] = v[nx * iy + 1];
    }
}

double
compute_error(double *v, double *vp, int nx, int ny) {
  double w = 0.0;
  double e = 0.0;
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
          int    lin_idx = nx * iy + ix;
          double d       = fabs(v[lin_idx] - vp[lin_idx]);
          e              = (d > e) ? d : e;
          w += fabs(v[lin_idx]);
        }
    }

  w /= (nx * ny);
  e /= w;

  return e;
}

int
solver(double *v, double *f, int nx, int ny, double eps, int nmax) {
  int     n = 0;
  double  e = 2. * eps;
  double *vp;

  // temp approximate solution vector
  vp = (double *)malloc(nx * ny * sizeof(double));

    // loop until convergence
    while ((e > eps) && (n < nmax)) { // step k
        // max difference between two consecutive iterations
        for (int i = 0; i < SUB_ITER / 2; i++) {
          solve(v, vp, f, nx, ny);
          // vp contains the updated solution
          apply_boundary(vp, nx, ny);
          solve(vp, v, f, nx, ny);
          // v contains the updated solution
          apply_boundary(v, nx, ny);
        }

      e = compute_error(v, vp, nx, ny);
      n += SUB_ITER;
    }

  free(vp);

  if (e < eps)
    printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e);
  else
    printf("ERROR: Failed to converge\n");

  return (e < eps ? 0 : 1);
}