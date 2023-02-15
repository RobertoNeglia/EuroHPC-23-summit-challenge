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
#include <mpi.h>

#ifndef NX
#  define NX 128
#endif
#ifndef NY
#  define NY 128
#endif
#define NMAX 200000
#define EPS 1e-8

int
solver(double *, double *, int, int, double, int, int, int);

int
main(int argc, char** argv) {
  double *v;
  double *f;

  //Initialisation of MPI
  MPI_Init(&argc, &argv);
	int rank = 0;
	int p = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Allocate memory
  // approximate solution vector
  int size = ceil(NX * NY * sizeof(double) / 2);
  v = (double *)malloc(size);
  // forcing term
  f = (double *)malloc(size);

  printf("Matrix size: %d\n", NX * NY);

  // Initialize input
    for (int iy = 0; iy < ceil(NY/2); iy++) {
        for (int ix = 0; ix < ceil(NX/2); ix++) {
          // initial guess is 0
          v[NX * iy + ix] = 0.0;

          const double x = 2.0 * (2 * ix + rank) / (NX - 1.0) - 1.0;
          const double y = 2.0 * (2 * iy + rank) / (NY - 1.0) - 1.0;
          // forcing term is a sinusoid
          f[NX * iy + ix] = sin(x + y);
        }
    }

  const clock_t start = clock();
  // Call solver
  solver(v, f, NX, NY, EPS, NMAX, p, rank);
  const clock_t end = clock();

  //Writing the results
  if (rank == 0) {
    const double dt = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time elapsed: %f[ms]\n", dt);

    // prints the approximate solution
    // for (int iy = 0; iy < NY; iy++)
    //   for (int ix = 0; ix < NX; ix++)
    //     printf("%d,%d,%e\n", ix, iy, v[iy * NX + ix]);

    char *filename = "solution_mpi.csv";

    FILE *file = fopen(filename, "w");
    fprintf(file, "x,y,v\n");
    for (int iy = 0; iy < NY; iy++)
      for (int ix = 0; ix < NX; ix++)
        fprintf(file, "%d,%d,%lf\n", ix, iy, v[iy * NX + ix]);
    fclose(file);

    printf("Output written to %s\n", filename);
  }
  // Clean-up
  free(v);
  free(f);
  MPI_Finalize();

  return 0;
}