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

#ifndef BLOCK_SIZE
#  define BLOCK_SIZE 32
#endif

// __global__ void
// solver(const double *v, const double *f, const int nx, const int ny, double *vp) {
//   const int tx = blockIdx.x * blockDim.x + threadIdx.x + 1;
//   const int ty = blockIdx.y * blockDim.y + threadIdx.y + 1;

//     if ((tx < (nx - 1)) && (ty < (ny - 1))) {
//       const int lin_idx = ty * nx + tx;
//       vp[lin_idx]                  // v_{i,j}
//         = -0.25                    // 1/t_{i,i} --> this is D^-1
//         * (f[lin_idx]              // f_{i,j}
//            - (v[lin_idx + 1]       // v_{i+1,j} -->
//               + v[lin_idx - 1]     // v_{i-1,j} --> this is R*v
//               + v[lin_idx + nx]    // v_{i,j+1} -->
//               + v[lin_idx - nx])); // v_{i,j-1} -->
//   }
// }

__global__ void
solver(const double *v, const double *f, const int nx, const int ny, double *vp) {
  // thread coordinates
  const int tx = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y + 1;

  __shared__ double vs[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // load data in shared memory
    for (int i = threadIdx.y; i < BLOCK_SIZE + 2; i += BLOCK_SIZE) {
        for (int j = threadIdx.x; j < BLOCK_SIZE + 2; j += BLOCK_SIZE) {
          const int tx = blockIdx.x * blockDim.x + j;
          const int ty = blockIdx.y * blockDim.y + i;

          double val = 0.0;

          if (tx < nx && ty < ny)
            val = v[ty * nx + tx];

          vs[i][j] = val;
        }
    }

  __syncthreads();

  // boundary check
    if ((tx < (nx - 1)) && (ty < (ny - 1))) {
      // linearize coordinates
      const int lin_idx = ty * nx + tx;
      // update solution
      vp[lin_idx]                                     // v_{i,j}
        = -0.25                                       // 1/t_{i,i} --> this is D^-1
        * (f[lin_idx]                                 // f_{i,j}
           - (vs[threadIdx.y + 0][threadIdx.x + 1]    // v[lin_idx + 1] v_{i+1,j} -->
              + vs[threadIdx.y + 2][threadIdx.x + 1]  // v[lin_idx - 1] v_{i-1,j} -->thisisR*v
              + vs[threadIdx.y + 1][threadIdx.x + 2]  // v[lin_idx + nx] v_{i,j+1} -->
              + vs[threadIdx.y + 1][threadIdx.x + 0]) // v[lin_idx - nx])); v_{i,j-1} -->
          );
  }
}

__global__ void
apply_bcs(double *v, const int nx, const int ny) {
  // thread coordinates
  const int tx = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (blockIdx.y == 0) {
        if (tx < (nx - 1)) {
          // y = 0
          v[tx] = v[nx * (ny - 2) + tx];
        } else {
          const int ty = tx - (nx - 2);
            if (ty < (ny - 1)) {
              // x = NX
              v[nx * ty + (nx - 1)] = v[nx * ty + 1];
          }
        }
    } else {
        if (tx < (nx - 1)) {
          // y = NY
          v[nx * (ny - 1) + tx] = v[nx + tx];
        } else {
          const int ty = tx - (nx - 2);
            if (ty < (ny - 1)) {
              // x = 0
              v[nx * ty] = v[nx * ty + (nx - 2)];
          }
        }
    }
}

__global__ void
compute_error(const double *v, const double *vp, const int size, double *es, double *ws) {
  __shared__ double reduce_max_error[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ double reduce_weight[BLOCK_SIZE * BLOCK_SIZE];

  const int t   = threadIdx.x;
  const int tid = blockIdx.x * blockDim.x + t;

  reduce_max_error[t] = 0;
  reduce_weight[t]    = 0;

    if (tid < size) {
      const double d      = fabs(vp[tid] - v[tid]);
      const double w      = fabs(v[tid]);
      reduce_max_error[t] = d;
      reduce_weight[t]    = w;
  }

  int active_elements = blockDim.x;

    while (active_elements > 1) {
      __syncthreads();
      active_elements /= 2;
        if (t < active_elements) {
          reduce_weight[t] += reduce_weight[t + active_elements];
          reduce_max_error[t] = (reduce_max_error[t + active_elements] > reduce_max_error[t]) ?
                                  reduce_max_error[t + active_elements] :
                                  reduce_max_error[t];
      }
    }

    if (t == 0) {
      ws[blockIdx.x] = reduce_weight[t];
      es[blockIdx.x] = reduce_max_error[t];
  }
}
