#include <assert.h>
#include <iostream>

/*
  Tiled 2d Convolution implementation using shared and constant memory
  @param A : input matrix
  @param widht : width of A
  @param height : height of A
  @param F : Filter
  @param MASK_SIZE : dimension of F
*/

const int TILE_SIZE = 16;
const int MASK_SIZE = 5;

__global__ void convolution_2d_tiled(float *A, int width, 
                                     int height, float *F, float *P) {

  int const MASK_OFFSET = MASK_SIZE/2;

  __shared__ float i_shmem[TILE_SIZE + MASK_SIZE-1][TILE_SIZE + MASK_SIZE-1];

  // local position of thread inside the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // boundaries
  int o_row = blockIdx.x * TILE_SIZE + tx;
  int o_col = blockIdx.y * TILE_SIZE + ty;

  // position of the thread in the input space
  int i_row = o_row - MASK_OFFSET;
  int i_col = o_col - MASK_OFFSET;

  if ((i_row >= 0 && i_row < height)
  && (i_col >= 0 && i_col < width))
    i_shmem[tx][ty] = A[i_row * width + i_col];
  else
    i_shmem[tx][ty] = 0.0f;

  // wait that all threads have loaded into shmem
  __syncthreads();

  float tmp = 0.0f;

    if (tx < TILE_SIZE && ty < TILE_SIZE) {
      // compute output element
      for (int i = 0; i < MASK_SIZE; i++) {
        for (int j = 0; j < MASK_SIZE; j++) {
          tmp += F[i*MASK_SIZE+j] * i_shmem[i+ty][j+tx];
        }
      }

      if(o_row < height && o_col < width){
        P[o_row * width + o_col] = tmp;
      }
    }
}

int main () {

  const int N = 16;

  float A[N][N], C[N][N], F[MASK_SIZE][MASK_SIZE];
  float *c_A, *c_F, *c_C;

  int a_size = sizeof(float) * N * N;
  int f_size = sizeof(float) * MASK_SIZE * MASK_SIZE;

  // allocate memory on device
  cudaMalloc((void**)&c_A,a_size);
  cudaMalloc((void**)&c_F,f_size);
  cudaMalloc((void**)&c_C,a_size);

  for(int j=0; j < N; j++) {
    for(int k=0; k < N; k++) { A[j][k] = 1; C[j][k] = 0; }}

  for(int j=0; j < MASK_SIZE; j++) {
          for(int k=0; k < MASK_SIZE; k++) { F[j][k] = 1; }}

  // copy from host
  cudaMemcpy(c_A,&A,a_size,cudaMemcpyHostToDevice);
  cudaMemcpy(c_F,&F,f_size,cudaMemcpyHostToDevice);
  cudaMemcpy(c_C,&C,a_size,cudaMemcpyHostToDevice);

  dim3 block(TILE_SIZE + MASK_SIZE - 1,TILE_SIZE + MASK_SIZE - 1);
  dim3 grid(N/TILE_SIZE + 1, N/TILE_SIZE + 1);

  convolution_2d_tiled<<<grid,block>>>(c_A, N, N, c_F, c_C);

  // transfer back on host from device
  cudaMemcpy(C,c_C,a_size,cudaMemcpyDeviceToHost);

  for(int j=0; j < N; j++) {
    for(int k=0; k < N; k++) {
      std::cout << C[j][k] << " ";
    }
    std::cout << std::endl;
  }


}