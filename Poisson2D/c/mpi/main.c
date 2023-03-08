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
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#ifndef NX
#  define NX 128
#endif
#ifndef NY
#  define NY 128
#endif
#define NMAX 100000
#define EPS 1e-8

int
solver(double *, double *, int, int, int, int, double, int, int, int*, int*, int*, MPI_Comm);

// void optimal_solution(int p, int* ptr_x, int* ptr_y, int* x_rank, int* y_rank) {
//   if (NX >= p * NY) {
//     *ptr_x = ceil(NX /p);
//     *ptr_y = NY;
//     return;
//   }
//   if (NY >= p * NX) {
//     *ptr_x = NX;
//     *ptr_y = ceil(NY / p);
//     return;
//   }
//   *ptr_x = NX / ceil(p/2);
//   *ptr_y = NY / (p/2);
//   return;
// }

int main(int argc, char** argv) {
  double *v;
  double *f;

  //Initialisation of MPI
  MPI_Init(&argc, &argv);
	int rank = 0;
	int p = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &p);

  // We divide the processors in a cart in 2 dimensions

  // optimal_solution(dims);
  int dims[2] = {0, 0};
  MPI_Dims_create(p, 2, dims);

  int periods[2] = {true, true};
  int reorder = true;
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);
  MPI_Comm_rank(cart, &rank);

  int coords[2];
  MPI_Cart_coords(cart, rank, 2, coords);
  
  // Allocate memory
  // approximate solution vector
  //We add a row/column at eache side to handle the boundary
  int size[] = {ceil((NX - 2)/dims[0]) + 2, ceil((NY - 2)/dims[1]) + 2};
  v = (double *)malloc(size[0] * size[1] * sizeof(double));
  // forcing term
  f = (double *)malloc(size[0] * size[1] * sizeof(double));

  int offset[] = {(size[0] - 2) * coords[0], (size[1] - 2) * coords[1]};

  printf("Matrix size: %d\n", NX * NY);

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

  const clock_t start = clock();
  // Call solver
  solver(v, f, NX, NY, size[0], size[1], EPS, NMAX, rank, coords, offset, dims, cart);
  const clock_t end = clock();

  //Writing the results
  if (rank == 0) {
    const double dt = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time elapsed: %f[ms]\n", dt);
  }

    // prints the approximate solution
    // for (int iy = 0; iy < NY; iy++)
    //   for (int ix = 0; ix < NX; ix++)
    //     printf("%d,%d,%e\n", ix, iy, v[iy * NX + ix]);

    char *filename = "solution_mpi.csv";
    int access_mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
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
    if (rank == 0) printf("Output written to %s\n", filename);

  // Clean-up
  free(v);
  free(f);
  MPI_Finalize();

  return 0;
}