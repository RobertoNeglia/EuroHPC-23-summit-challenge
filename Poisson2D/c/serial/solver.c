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

int
solver(double *v, double *f, int nx, int ny, double eps, int nmax) {
  int     n = 0;
  double  e = 2. * eps;
  double *v_copy, *v_old, *v_new, *swap;

  // temp approximate solution vector
  v_copy = (double *)malloc(nx * ny * sizeof(double));

  v_old = &v[0]; // old
  v_new = &v_copy[0];

    // loop until convergence
    while ((e > eps) && (n < nmax)) { // step k
      // max difference between two consecutive iterations
      e = 0.0;

      double w = 0.0;

      // loop over each element of the discretized domain
        for (int iy = 1; iy < (ny - 1); iy++) {
            for (int ix = 1; ix < (nx - 1); ix++) {
              double d;

              // compute v^{k+1}
              v_new[iy * nx + ix]                    // v_{i,j}
                = -0.25                              // 1/t_{i,i} --> this is D^-1
                * (f[iy * nx + ix]                   // f_{i,j}
                   - (v_old[nx * iy + ix + 1]        // v_{i+1,j} -->
                      + v_old[nx * iy + ix - 1]      // v_{i-1,j} --> this is R*v
                      + v_old[nx * (iy + 1) + ix]    // v_{i,j+1} -->
                      + v_old[nx * (iy - 1) + ix])); // v_{i,j-1} -->

              // compute difference between iteration k and k-1
              d = fabs(v_new[nx * iy + ix] - v_old[nx * iy + ix]);
              e = (d > e) ? d : e;
              w += fabs(v_new[nx * iy + ix]);
            }
        }
      // swap pointers
      swap  = v_new;
      v_new = v_old;
      v_old = swap;
      // now inside v_old there is the updated solution

      // Update v and compute error as well as error weight factor

        // compute weight on boundaries & apply boundary conditions
        for (int ix = 1; ix < (nx - 1); ix++) {
          // y = 0
          v_old[nx * 0 + ix] = v_old[nx * (ny - 2) + ix];
          // y = NY
          v_old[nx * (ny - 1) + ix] = v_old[nx * 1 + ix];
          w += fabs(v_old[nx * 0 + ix]) + fabs(v_old[nx * (ny - 1) + ix]);
        }

        for (int iy = 1; iy < (ny - 1); iy++) {
          // x = 0
          v_old[nx * iy + 0] = v_old[nx * iy + (nx - 2)];
          // x = NX
          v_old[nx * iy + (nx - 1)] = v_old[nx * iy + 1];
          w += fabs(v_old[nx * iy + 0]) + fabs(v_old[nx * iy + (nx - 1)]);
        }

      // update weight by domain size
      w /= (nx * ny);
      // update difference of consecutive iterations
      e /= w;

      // if ((n % 10) == 0)
      //     printf("%5d, %0.4e\n", n, e);

      n++;
    }

    if (n % 2) {
      v      = v_new;
      v_copy = v_old;
    } else {
      v      = v_old;
      v_copy = v_new;
    }

  free(v_copy);

  if (e < eps)
    printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e);
  else
    printf("ERROR: Failed to converge\n");

  return (e < eps ? 0 : 1);
}