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

__global__ void
solver(double *v, double *f, int nx, int ny, double *vp) {
  int tx      = blockIdx.x * blockDim.x + threadIdx.x;
  int ty      = blockIdx.y * blockDim.y + threadIdx.y;
  int stridex = blockDim.x * gridDim.x;
  int stridey = blockDim.y * gridDim.y;

    for (int iy = ty + 1; iy < ny - 1; iy += stridey) {
        for (int ix = tx + 1; ix < nx - 1; ix += stridex) { // compute v^{k+1}
          vp[iy * nx + ix]                                  // v_{i,j}
            = -0.25                                         // 1/t_{i,i} --> this is D^-1
            * (f[iy * nx + ix]                              // f_{i,j}
               - (v[nx * iy + ix + 1]                       // v_{i+1,j} -->
                  + v[nx * iy + ix - 1]                     // v_{i-1,j} --> this is R*v
                  + v[nx * (iy + 1) + ix]                   // v_{i,j+1} -->
                  + v[nx * (iy - 1) + ix]));                // v_{i,j-1} -->
        }
    }
}

__global__ void
apply_boundary(double *v, int nx, int ny) {
  int tx      = blockIdx.x * blockDim.x + threadIdx.x;
  int ty      = blockIdx.y * blockDim.y + threadIdx.y;
  int stridex = blockDim.x * gridDim.x;
  int stridey = blockDim.y * gridDim.y;

    // compute weight on boundaries & apply boundary conditions
    for (int ix = tx + 1; ix < (nx - 1); ix += stridex) {
      // y = 0
      v[ix] = v[nx * (ny - 2) + ix];
      // y = NY
      v[nx * (ny - 1) + ix] = v[nx + ix];
    }

    for (int iy = ty + 1; iy < (ny - 1); iy += stridey) {
      // x = 0
      v[nx * iy] = v[nx * iy + (nx - 2)];
      // x = NX
      v[nx * iy + (nx - 1)] = v[nx * iy + 1];
    }
}