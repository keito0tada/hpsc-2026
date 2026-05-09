#include <cstdio>
#include <cstdlib>

__global__ void count_kernel(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

int main() {
  int n = 50;
  int range = 5;
  int M = 128;

  int *key, *bucket;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  for (int i = 0; i < range; i++) bucket[i] = 0;

  count_kernel<<<(n+M-1)/M, M>>>(key, bucket, n);
  cudaDeviceSynchronize();

  for (int i = 0, j = 0; i < range; i++) {
    for (; bucket[i] > 0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}

