#include <assert.h>
#include <iostream>

// baseline 2d convolution
// Only use odd kernel sizes
__global__ void convolution_2d(int *A,int *F, int p, int n, int *C) {

  int tmp = 0;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.x*blockDim.x + tx;
  int col = blockIdx.y*blockDim.y + ty;

  int offset_k = p/2;

  int row_offset = row - offset_k;
  int col_offset = col - offset_k;

  for(int kx = 0; kx < p; kx++) {
    for(int ky = 0; ky < p; ky++) {
      if(row_offset + kx >= 0 && row_offset + kx < n) {
        if(col_offset + ky >= 0 && col_offset + ky < n)
          tmp += A[(row_offset + kx)*n + col_offset + ky]
                * F[kx * p + ky];
      }
    }
  }

  if(row < n && col < n){
    C[row*n + col] = tmp;
  }
}

int main () {

  const int WARP_SIZE = 32;

  int N = 1024;

  const int MASK_SIZE = 3;
  const int BLOCK_SIZE = 32;

  int *A,*C,*F;
  int *c_A, *c_F, *c_C;

  int a_size = sizeof(int) * N * N;
  int f_size = sizeof(int) * MASK_SIZE * MASK_SIZE;

  A = (int*)malloc(a_size);
  F = (int*)malloc(f_size);
  C = (int*)malloc(a_size);

  // allocate memory on device
  cudaMalloc((void**)&c_A,a_size);
  cudaMalloc((void**)&c_F,f_size);
  cudaMalloc((void**)&c_C,a_size);

  for(int j = 0; j < N; j++) {
    for(int k = 0; k < N; k++) {
      A[j*N+k] = 1;
      C[j*N+k] = 0;
    }
  }

  for(int j=0; j < MASK_SIZE; j++) {
    for(int k=0; k < MASK_SIZE; k++) {
      F[j*MASK_SIZE+k] = 1;
    }
  }

  cudaMemcpy(c_A,&A,a_size,cudaMemcpyHostToDevice);
  cudaMemcpy(c_F,&F,f_size,cudaMemcpyHostToDevice);
  cudaMemcpy(c_C,&C,a_size,cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid(N/BLOCK_SIZE+1,N/BLOCK_SIZE+1);

  convolution_2d<<<grid,block>>>(c_A, c_F, MASK_SIZE, N, c_C);

  // transfer back on host from device
  cudaMemcpy(C,c_C,a_size,cudaMemcpyDeviceToHost);

  free(A);
  free(F);
  free(C);

  cudaFree(c_A);
  cudaFree(c_F);
  cudaFree(c_C);

}