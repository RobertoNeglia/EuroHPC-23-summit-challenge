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
#include <mpi.h>

int solver(double *v, double *f, int NX, int NY, int nx, int ny, double eps, int nmax, int rank, int* coords, int* offset, int* dims, MPI_Comm cart) {
  int     n = 0;
  double  e = 2. * eps;
  double *v_copy, *v_old, *v_new, *swap;

  // temp approximate solution vector
  v_copy = (double *)malloc(nx * ny * sizeof(double));

  v_old = &v[0]; // old
  v_new = &v_copy[0];

  int up, down, left, right;
  MPI_Cart_shift(cart, 0, 1, &up, &down);
  MPI_Cart_shift(cart, 1, 1, &left, &right);

  int x_start_ind = 1;
  int y_start_ind = 1;
  int x_end_ind   = nx - 1;
  int y_end_ind   = ny - 1;

  MPI_Datatype x_type;
  MPI_Type_vector(1, nx, 1, MPI_DOUBLE, &x_type);
  MPI_Type_commit(&x_type);

  MPI_Datatype y_type;
  MPI_Type_vector(ny, 1, nx, MPI_DOUBLE, &y_type);
  MPI_Type_commit(&y_type);


  int up_send_ind    = x_start_ind;
  int up_recv_ind    = x_start_ind - 1;
  int down_send_ind  = x_end_ind - 1;
  int down_recv_ind  = x_end_ind;
  int left_send_ind  = nx * y_start_ind;
  int left_recv_ind  = nx * (y_start_ind - 1);
  int right_send_ind = nx * (y_end_ind - 1);
  int right_recv_ind = nx * y_end_ind;

  // int odd = (coords[0] + coords[1]) % 2;

  // loop until convergence
  while ((e > eps) && (n < nmax)) { // step k
    // max difference between two consecutive iterations
    e = 0.0;

    double w = 0.0;

    // loop over each element of the discretized domain
    for (int iy = y_start_ind; iy < y_end_ind; iy++) {
      for (int ix = x_start_ind; ix < x_end_ind; ix++) {
        double d;

        // compute v^{k+1}
        v_new[iy * nx + ix]                    // v_{i,j}
          = -0.25                              // 1/t_{i,i} --> this is D^-1
          * (f[iy * nx + ix]                   // f_{i,j}
              - (v_old[nx * iy + ix + 1]       // v_{i+1,j} --> down
                + v_old[nx * iy + ix - 1]      // v_{i-1,j} --> up
                + v_old[nx * (iy + 1) + ix]    // v_{i,j+1} --> right
                + v_old[nx * (iy - 1) + ix])); // v_{i,j-1} --> left

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

    // compute weight on boundaries & apply boundary conditions
    // if (odd) {
    //   MPI_Sendrecv(&v_old[up_send_ind], 1, x_type, up, n, &v_old[up_recv_ind], 1, x_type, up, n, cart, MPI_STATUS_IGNORE);
    //   MPI_Sendrecv(&v_old[left_send_ind], 1, y_type, left, n * 2, &v_old[left_recv_ind], 1, y_type, left, n * 2, cart, MPI_STATUS_IGNORE);
    // }
    // MPI_Sendrecv(&v_old[down_send_ind], 1, x_type, down, n, &v_old[down_recv_ind], 1, x_type, down, n, cart, MPI_STATUS_IGNORE);
    // MPI_Sendrecv(&v_old[right_send_ind], 1, y_type, right, n * 2, &v_old[right_recv_ind], 1, y_type, right, n * 2, cart, MPI_STATUS_IGNORE);
    // MPI_Sendrecv(&v_old[up_send_ind], 1, x_type, up, n, &v_old[up_recv_ind], 1, x_type, up, n, cart, MPI_STATUS_IGNORE);
    // MPI_Sendrecv(&v_old[left_send_ind], 1, y_type, left, n * 2, &v_old[left_recv_ind], 1, y_type, left, n * 2, cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&v_old[left_send_ind], 1, x_type, left, 1, &v_old[right_recv_ind], 1, x_type, right, 1, cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&v_old[right_send_ind], 1, x_type, right, 1, &v_old[left_recv_ind], 1, x_type, left, 1, cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&v_old[up_send_ind], 1, y_type, up, 1, &v_old[down_recv_ind], 1, y_type, down, 1, cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&v_old[down_send_ind], 1, y_type, down, 1, &v_old[up_recv_ind], 1, y_type, up, 1, cart, MPI_STATUS_IGNORE);


    // update weight by domain size 
    // This part can take a lot of time if the matrix is bigger than the L1 cache 
    // and it is not essential, we just have to take a smaller error
    if (coords[0] == dims[0] - 1)
      for (int ix = x_start_ind; ix < x_end_ind; ix++)
        w += fabs(v_old[nx * (y_start_ind - 1) + ix]);
        
    if (coords[1] == dims[1] - 1)
      for (int ix = x_start_ind; ix < x_end_ind; ix++)
        w += fabs(v_old[nx * y_end_ind + ix]);

    if (coords[0] == 0)
      for (int iy = y_start_ind; iy < y_end_ind; iy++) 
         w += fabs(v_old[nx * iy + x_start_ind - 1]);

    if (coords[1] == 0)
      for (int iy = y_start_ind; iy < y_end_ind; iy++)
        w += fabs(v_old[nx * iy + x_end_ind]);



    w /= NX * NY;
    MPI_Allreduce(MPI_IN_PLACE, &w, 1, MPI_DOUBLE, MPI_SUM, cart);
    // update difference of consecutive iterations
    e /= w;
    MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_DOUBLE, MPI_MAX, cart);

    // if ((n % 100) == 0) printf("%5d, %0.4e\n", n, e);

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