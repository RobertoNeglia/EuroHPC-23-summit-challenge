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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SUB_ITER 100

void
allocate_on_device(double   *v_dev,
                   double   *vp_dev,
                   double   *f_dev,
                   double   *e_dev,
                   double   *w_dev,
                   const int nx,
                   const int ny,
                   double   *diff_size);

void
free_on_device(double *v_dev, double *vp_dev, double *f_dev, double *e_dev, double *w_dev);

void
solve_on_device(double       *,
                const double *,
                double       *,
                const double *,
                const int     ,
                const int     ,
                double       *,
                double       *);

void
compute_error_on_device(const double *,
                        const double *,
                        const int     ,
                        const int     ,
                        double       *,
                        double       *,
                        double       *,
                        double       *);


int
solver(double  *v,
       double  *f,
       int      NX,
       int      NY,
       int      nx,
       int      ny,
       double   eps,
       int      nmax,
       int      rank,
       int     *coords,
       int     *offset,
       int     *dims,
       MPI_Comm cart) {
  double *vp, *e, *w;
  double *v_dev = NULL, *vp_dev= NULL, *f_dev= NULL, *e_dev= NULL, *w_dev= NULL;
  double  diff_size;

  // allocate vectors on the device
  allocate_on_device(v_dev, vp_dev, f_dev, e_dev, w_dev, nx, ny, &diff_size);

  // temp host vectors
  vp = (double *)malloc(nx * ny * sizeof(double));
  e  = (double *)malloc(diff_size * sizeof(double));
  w  = (double *)malloc(diff_size * sizeof(double));

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

  int     n = 0;
  double e_max = 2. * eps;
  double w_sum   = 0.0;
    // loop until convergence
    while ((e_max > eps) && (n < nmax)) {
        for (int i = 0; i < SUB_ITER / 2; i++) {
          solve_on_device(v_dev, v, f_dev, f, nx, ny, vp_dev, vp);
          // vp has the updated solution, send it to neighbouring nodes
          // left node
          MPI_Sendrecv(&vp[left_send_ind],
                       1,
                       x_type,
                       left,
                       1,
                       &vp[right_recv_ind],
                       1,
                       x_type,
                       right,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
          // right node
          MPI_Sendrecv(&vp[right_send_ind],
                       1,
                       x_type,
                       right,
                       1,
                       &vp[left_recv_ind],
                       1,
                       x_type,
                       left,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
          // up node
          MPI_Sendrecv(&vp[up_send_ind],
                       1,
                       y_type,
                       up,
                       1,
                       &vp[down_recv_ind],
                       1,
                       y_type,
                       down,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
          // down node
          MPI_Sendrecv(&vp[down_send_ind],
                       1,
                       y_type,
                       down,
                       1,
                       &vp[up_recv_ind],
                       1,
                       y_type,
                       up,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);

          solve_on_device(vp_dev, vp, f_dev, f, nx, ny, v_dev, v);
          // v has the updated solution, send it to neighbouring nodes
          // left node
          MPI_Sendrecv(&v[left_send_ind],
                       1,
                       x_type,
                       left,
                       1,
                       &v[right_recv_ind],
                       1,
                       x_type,
                       right,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
          // right node
          MPI_Sendrecv(&v[right_send_ind],
                       1,
                       x_type,
                       right,
                       1,
                       &v[left_recv_ind],
                       1,
                       x_type,
                       left,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
          // up node
          MPI_Sendrecv(&v[up_send_ind],
                       1,
                       y_type,
                       up,
                       1,
                       &v[down_recv_ind],
                       1,
                       y_type,
                       down,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
          // down node
          MPI_Sendrecv(&v[down_send_ind],
                       1,
                       y_type,
                       down,
                       1,
                       &v[up_recv_ind],
                       1,
                       y_type,
                       up,
                       1,
                       cart,
                       MPI_STATUS_IGNORE);
        }
      // big reduce on device
      compute_error_on_device(v_dev, vp_dev, nx, ny, e_dev, e, w_dev, w);
      // finish the reduce on host
      e_max = 0.0;
      w_sum = 0.0;
      for (int i = 0; i < diff_size; i++) {
          e_max = (e[i] > e_max) ? e[i] : e_max;
          w_sum += w[i];
        }

      w_sum /= NX * NY;
      MPI_Allreduce(MPI_IN_PLACE, &w_sum, 1, MPI_DOUBLE, MPI_SUM, cart);
      e_max /= w_sum;
      MPI_Allreduce(MPI_IN_PLACE, &e_max, 1, MPI_DOUBLE, MPI_MAX, cart);

      n+=SUB_ITER;
    }

  free_on_device(v_dev, vp_dev, f_dev, e_dev, w_dev);

  free(vp);
  free(e);
  free(w);

  if (!rank)
    if (e_max < eps)
      printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e_max);
    else
      printf("ERROR: Failed to converge\n");

  return (e_max < eps ? 0 : 1);
}