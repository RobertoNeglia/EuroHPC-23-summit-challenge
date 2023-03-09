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
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NX
#  define NX 2048
#endif
#ifndef NY
#  define NY 2048
#endif
#define NMAX 200000
#define EPS 1e-5

#ifndef max
#  define max(a,b) (((a)>(b))?(a):(b))
#endif

int
solver(double *,
       double *,
       int,
       int,
       int,
       int,
       double,
       int,
       int,
       int *,
       int *,
       int *,
       MPI_Comm);

void optimal_dimension(int p, int* dims, int nx, int ny) {
  int min_length = nx;
  int y, length;
  for (int x = 1; x <= p; x++) {
    if (p % x == 0) {
      y = p / x;
      // We compute the greatest communication for each dimension, 
      // taking in account the possibility that the matrix cannot be equally shared
      int x_length = (nx - 2)/x + nx - 2 - (nx - 2)/x * x;
      int y_length = (ny - 2)/y + ny - 2 - (ny - 2)/y * y;
      length = max(x_length, y_length);
      if (length <= min_length) {
        min_length = length;
        dims[0] = x;
        dims[1] = y;
      }
    }
  }
  return;
}

int
main(int argc, char **argv) {
  double *v;
  double *f;

  // Initialisation of MPI
  MPI_Init(&argc, &argv);
  int rank = 0;
  int p    = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // We divide the processors in a cart in 2 dimensions  
  int dims[2];
  // This function creates a cart such that each processor
  // has a matrix as squared as possible to avoid too much communication cost.
  optimal_dimension(p, dims, NX, NY);
  MPI_Dims_create(p, 2, dims);

  int      periods[2] = {true, true};
  int      reorder    = true;
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);
  MPI_Comm_rank(cart, &rank);

  int coords[2];
  MPI_Cart_coords(cart, rank, 2, coords);

  // Allocate memory
  // approximate solution vector
  // We add a row/column at eache side to handle the boundary
  int size[]   = {((NX - 2) / dims[0]) + 2, ((NY - 2) / dims[1]) + 2};
  int offset[] = {(size[0] - 2) * coords[0], (size[1] - 2) * coords[1]};

  // If the matrix v cannot be equally shared, we adjust it at the end of the cart
  if (dims[0] - 1 == coords[0]) size[0] += (NX - 2 - (size[0] - 2) * dims[0]);

  if (dims[1] - 1 == coords[1]) size[1] += (NY - 2 - (size[1] - 2) * dims[1]);


  v            = (double *)malloc(size[0] * size[1] * sizeof(double));

  // forcing term
  f = (double *)malloc(size[0] * size[1] * sizeof(double));


  if (!rank)
    {
      printf("Matrix size: %d\n", NX * NY);
      printf("nodes: %d\n", p);
    }

  // Initialize input
    for (int iy = 0; iy < size[1]; iy++) {
        for (int ix = 0; ix < size[0]; ix++) {
          // initial guess is 0
          v[size[0] * iy + ix] = 0.0;

          const double x = 2.0 * (ix + offset[0]) / (NX - 1.0) - 1.0;
          const double y = 2.0 * (iy + offset[1]) / (NY - 1.0) - 1.0;
          // forcing term is a sinusoid
          f[size[0] * iy + ix] = sin(x + y);
        }
    }

  const double start = omp_get_wtime();
  // Call solver
  solver(v, f, NX, NY, size[0], size[1], EPS, NMAX, rank, coords, offset, dims, cart);
  const double end = omp_get_wtime();

    // Writing the results
    if (rank == 0) {
      const double dt = (end - start) * 1000;
      printf("Time elapsed: %f[ms]\n", dt);
  }

  // prints the approximate solution
  // for (int iy = 0; iy < NY; iy++)
  //   for (int ix = 0; ix < NX; ix++)
  //     printf("%d,%d,%e\n", ix, iy, v[iy * NX + ix]);

  char    *filename    = "solution_mpi.csv";
  int      access_mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, filename, access_mode, MPI_INFO_NULL, &file);
  char buf[42];
  snprintf(buf, 42, "x,y,v\n");
  MPI_File_write(file, buf, strlen(buf), MPI_CHAR, MPI_STATUS_IGNORE);
  for (int iy = 0; iy < size[1]; iy++)
      for (int ix = 0; ix < size[0]; ix++) {
        snprintf(buf, 42, "%d,%d,%lf\n", ix + offset[0], iy + offset[1], v[iy * size[0] + ix]);
        MPI_File_write(file, buf, strlen(buf), MPI_CHAR, MPI_STATUS_IGNORE);
      }
  MPI_File_close(&file);
  MPI_Barrier(cart);
  if (rank == 0)
    printf("Output written to %s\n", filename);

  // Clean-up
  free(v);
  free(f);
  MPI_Finalize();

  return 0;
}