#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NX
#  define NX 128
#endif
#ifndef NY
#  define NY 128
#endif

int
main(int argc, char **argv) {
  int          tmpx, tmpy;
  double      *v1, *v2;
  char         thrashx, thrashy, thrashv;
  const double tol = 1.e-2;
  char        *filename1, *filename2;

    if (argc > 3) {
      printf("ERROR: too many arguments\nCorrect usage: ./equal.out solution1.csv "
             "solution2.csv\n");
      return 0;
    } else if (argc < 3) {
      printf("ERROR: too few arguments\nCorrect usage: ./equal.out solution1.csv "
             "solution2.csv\n");
      return 0;
    } else {
      filename1 = argv[1];
      filename2 = argv[2];
    }

  v1 = (double *)malloc(NX * NY * sizeof(double));
  v2 = (double *)malloc(NX * NY * sizeof(double));

  FILE *sol1 = fopen(filename1, "r");
  fscanf(sol1, "%c,%c,%c\n", &thrashx, &thrashy, &thrashv);
  for (int iy = 0; iy < NY; iy++)
    for (int ix = 0; ix < NX; ix++)
      fscanf(sol1, "%d,%d,%lf\n", &tmpx, &tmpy, &v1[iy * NX + ix]);

  fclose(sol1);

  FILE *sol2 = fopen(filename2, "r");
  fscanf(sol2, "%c,%c,%c\n", &thrashx, &thrashy, &thrashv);
  for (int iy = 0; iy < NY; iy++)
    for (int ix = 0; ix < NX; ix++)
      fscanf(sol2, "%d,%d,%lf\n", &tmpx, &tmpy, &v2[iy * NX + ix]);

  fclose(sol2);

  double max_diff = -0.5;
  double val1, val2;
  int    x, y;

  for (int iy = 0; iy < NY; iy++)
      for (int ix = 0; ix < NX; ix++) {
          if (fabs(v1[iy * NX + ix] - v2[iy * NX + ix]) > max_diff) {
            max_diff = fabs(v1[iy * NX + ix] - v2[iy * NX + ix]);
            x        = ix;
            y        = iy;
            val1     = v1[iy * NX + ix];
            val2     = v2[iy * NX + ix];
        }
      }

  free(v1);
  free(v2);

  printf("The maximum difference between the two solution is: %lf\n", max_diff);
  printf("(%d, %d) -> v1=%lf; v2=%lf\n", x, y, val1, val2);

  return 0;
}