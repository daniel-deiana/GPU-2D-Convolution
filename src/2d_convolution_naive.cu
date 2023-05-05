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

  C[row*n + col] = tmp;
}

int main () {

  const int WARP_SIZE = 32;

  const int N = 4;
  const int P = 3;

  int A[N][N], C[N][N], F[P][P];
  int *c_A, *c_F, *c_C;

  int a_size = sizeof(int) *N*N;
  int f_size = sizeof(int) *P*P;

  // allocate memory on device
  cudaMalloc((void**)&c_A,a_size);
  cudaMalloc((void**)&c_F,f_size);
  cudaMalloc((void**)&c_C,a_size);

  for(int j=0; j < N; j++) {
    for(int k=0; k < N; k++) { A[j][k] = 1; C[j][k] = 0; }}

  for(int j=0; j < P; j++) {
          for(int k=0; k < P; k++) { A[j][k] = 1; }}

  cudaMemcpy(c_A,&A,a_size,cudaMemcpyHostToDevice);
  cudaMemcpy(c_F,&F,f_size,cudaMemcpyHostToDevice);
  cudaMemcpy(c_C,&C,a_size,cudaMemcpyHostToDevice);

  dim3 block(16,16);

  convolution_2d<<<1,block>>>(c_A, c_F, P, N, c_C);

  // transfer back on host from device
  cudaMemcpy(C,c_C,a_size,cudaMemcpyDeviceToHost);

  for(int j=0; j < N; j++) {
    for(int k=0; k < N; k++) {
      std::cout << C[j][k];
    }
    std::cout << std::endl;
  }


}
