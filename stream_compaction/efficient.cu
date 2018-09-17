#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "naive.h"
#include<cstdint>

#define blockSize 512

namespace StreamCompaction
{
  namespace Efficient
  {
    int* device_idata;
    int* device_bools;
    int* device_scannedBools;
    int* device_odata;
    int numObjects;

    void printArray(int n, int *a, bool abridged = false) {
      printf("    [ ");
      for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
          i = n - 2;
          printf("... ");
        }
        printf("%3d ", a[i]);
      }
      printf("]\n");
    }

    using StreamCompaction::Common::PerformanceTimer;

    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

    __global__ void kernel_UpSweep(int N, int powDP1, int* idata)
    {
      const int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= N) {
        return;
      }

      if (index % powDP1 != 0)
      {
        return;
      }

      // x[k + 2d+1 – 1] += x[k + 2d – 1];
      idata[index + powDP1 - 1] += idata[index + (powDP1 / 2) - 1];
    }

    __global__ void kernel_UpSweepOptimized_v1(int N, int numThreads, int powD, int* idata)
    {
      extern __shared__ int temp[];

      const int threadID = threadIdx.x;
      const int threadID2X = 2 * threadIdx.x;
      const int staticIdx = threadID2X + (blockIdx.x * N) + (powD - 1);

      int offset = 1;

      temp[threadID2X] = idata[staticIdx];
      temp[threadID2X + 1] = idata[staticIdx + powD];

      // build sum in place up the tree
      for (int d = numThreads; d > 0; d >>= 1)
      {
        __syncthreads();
        if (threadID < d)
        {
          const int ai = offset * (threadID2X + 1) - 1;
          const int bi = offset * (threadID2X + 2) - 1;
          temp[bi] += temp[ai];
        }
      
        offset *= 2;
      }

      __syncthreads();

      idata[staticIdx] = temp[threadID2X];
      idata[staticIdx + powD] = temp[threadID2X + 1];
    }

    __global__ void kernel_DownSweepOptimized_v1(int N, int numThreads, int powD, int* idata, int* odata)
    {
      extern __shared__ int temp[];

      const int threadID = threadIdx.x;
      const int threadID2X = 2 * threadIdx.x;
      const int staticIdx = threadID2X + (blockIdx.x * N) + (powD - 1);

      int offset = N / 2;

      temp[threadID2X] = idata[staticIdx];
      temp[threadID2X + 1] = idata[staticIdx + powD];

      // traverse down tree & build scan
      for (int d = 1; d <= numThreads; d *= 2)
      {
        offset >>= 1;
      
        __syncthreads();
      
        if (threadID < d)
        {
          const int ai = (threadID2X + 1) - 1;
          const int bi = (threadID2X + 2) - 1;
      
          const int t = temp[ai];
          temp[ai] = temp[bi];
          temp[bi] += t;
        }
      }

      __syncthreads();

      idata[staticIdx] = temp[threadID2X];
      idata[staticIdx + powD] = temp[threadID2X + 1];
    }

    __global__ void kernel_DownSweep(int N, int powDP1, int* idata)
    {
      const int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= N) {
        return;
      }

      if (index % powDP1 != 0)
      {
        return;
      }

      // Calculate some indices
      const int leftChildIdx = index + (powDP1 / 2) - 1;
      const int rightChildIdx = index + powDP1 - 1;

      // Save the left child
      const int leftChild = idata[leftChildIdx];

      // Set Left Child to Current Node's Value
      idata[leftChildIdx] = idata[rightChildIdx];

      // Set Right Child to Left + Right
      idata[rightChildIdx] += leftChild;
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int* odata, const int* idata)
    {
      numObjects = n;
      const int logN = ilog2ceil(numObjects);
      const int nearestPower2 = std::pow(2, logN);

      cudaMalloc((void**)&device_idata, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_idata failed!");

      cudaMemcpy(device_idata, idata, sizeof(int) * numObjects, cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy failed!");

      const int numBlocks = (numObjects + blockSize - 1) / blockSize;
      dim3 fullBlocksPerGrid(numBlocks);

      int* loopInputBuffer = device_idata;

      // Up Sweep
      timer().startGpuTimer();
      for (int d = 0; d < logN; ++d)
      {
        const int powDP1 = std::pow(2, d + 1);
        kernel_UpSweep<<<fullBlocksPerGrid, blockSize>>>(numObjects, powDP1, loopInputBuffer);
      }
      timer().endGpuTimer();

      // Set x[n-1] = 0
      // This seems really weird that we need to copy a 0 from host to the device.
      // Might need to find a more efficient way.
      const int lastValue = 0;
      cudaMemcpy(&loopInputBuffer[nearestPower2 - 1], &lastValue, sizeof(int), cudaMemcpyHostToDevice);
      
      // Down Sweep
      timer().startGpuTimer();
      for (int d = logN - 1; d >= 0; --d)
      {
        const int powDP1 = std::pow(2, d + 1);
        kernel_DownSweep<<<fullBlocksPerGrid, blockSize>>>(numObjects, powDP1, loopInputBuffer);
      }
      timer().endGpuTimer();

      cudaMemcpy(odata, loopInputBuffer, sizeof(int) * (numObjects), cudaMemcpyDeviceToHost);

      cudaFree(device_idata);
    }

    void scanOptimized_v1(int n, int* odata, const int* idata)
    {
      numObjects = n;
      const int logN = ilog2ceil(numObjects);
      const int nearestPower2 = std::pow(2, logN);

      cudaMalloc((void**)&device_idata, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_idata failed!");
      
      cudaMalloc((void**)&device_odata, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_idata failed!");

      cudaMemcpy(device_idata, idata, sizeof(int) * numObjects, cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy failed!");

      const int numBlocks = (numObjects + blockSize - 1) / blockSize;
      dim3 fullBlocksPerGrid(numBlocks);

      int* loopInputBuffer = device_idata;

      const int numCount = nearestPower2;

      int upSweepBlockCount = (numCount + blockSize - 1) / blockSize;
      const int downSweepBlockCount = (numCount + blockSize - 1) / blockSize;

      int depth = 0;

      // Up Sweep
      timer().startGpuTimer();
      while(upSweepBlockCount > 0)
      {
        const int powD = std::pow(2, depth);
        const int powD1 = std::pow(2, depth + 1);

        dim3 upSweepBlocks(upSweepBlockCount);

        const int numObjectsPerBlock = numCount / upSweepBlockCount;

        const int threadsPerBlock = numObjectsPerBlock / powD1;

        kernel_UpSweepOptimized_v1<<<upSweepBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(int)>>>(numObjectsPerBlock, threadsPerBlock, powD, loopInputBuffer);

        upSweepBlockCount /= 2;
        depth = ilog2ceil(numObjectsPerBlock);
      }
      timer().endGpuTimer();

      // Set x[n-1] = 0
      // This seems really weird that we need to copy a 0 from host to the device.
      // Might need to find a more efficient way.
      const int lastValue = 0;
      cudaMemcpy(&loopInputBuffer[nearestPower2 - 1], &lastValue, sizeof(int), cudaMemcpyHostToDevice);

      // Down Sweep
      timer().startGpuTimer();
      for (int d = logN - 1; d >= 0; --d)
      {
        const int powDP1 = std::pow(2, d + 1);
        kernel_DownSweep<<<fullBlocksPerGrid, blockSize>>>(numObjects, powDP1, loopInputBuffer);
      }
      timer().endGpuTimer();

      cudaMemcpy(odata, device_odata, sizeof(int) * (numObjects), cudaMemcpyDeviceToHost);

      cudaFree(device_idata);
    }

    /**
     * Performs stream compaction on idata, storing the result into odata.
     * All zeroes are discarded.
     *
     * @param n      The number of elements in idata.
     * @param odata  The array into which to store elements.
     * @param idata  The array of elements to compact.
     * @returns      The number of elements remaining after compaction.
     */
    int compact(int n, int* odata, const int* idata)
    {
      numObjects = n;
      const int logN = ilog2ceil(numObjects);
      const int nearestPower2 = std::pow(2, logN);

      cudaMalloc((void**)&device_idata, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_idata failed!");
      
      cudaMalloc((void**)&device_odata, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_odata failed!");

      cudaMalloc((void**)&device_bools, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_bools failed!");
      
      cudaMalloc((void**)&device_scannedBools, nearestPower2 * sizeof(int));
      checkCUDAError("cudaMalloc device_scannedBools failed!");

      cudaMemcpy(device_idata, idata, sizeof(int) * numObjects, cudaMemcpyHostToDevice);

      const int numBlocks = (numObjects + blockSize - 1) / blockSize;
      dim3 fullBlocksPerGrid(numBlocks);

      // 1. Get Bool Array 1st
      timer().startGpuTimer();
      Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(numObjects, device_bools, device_idata);
      cudaMemcpy(device_scannedBools, device_bools, sizeof(int) * nearestPower2, cudaMemcpyDeviceToDevice);

      // 2. Scan the Bool Array
      int* loopInputBuffer = device_scannedBools;

      // Up Sweep
      for (int d = 0; d < logN; ++d)
      {
        const int powDP1 = std::pow(2, d + 1);
        kernel_UpSweep<<<fullBlocksPerGrid, blockSize>>>(numObjects, powDP1, loopInputBuffer);
      }
      timer().endGpuTimer();

      // Set x[n-1] = 0
      // This seems really weird that we need to copy a 0 from host to the device.
      // Might need to find a more efficient way.
      const int lastValue = 0;
      cudaMemcpy(&loopInputBuffer[nearestPower2 - 1], &lastValue, sizeof(int), cudaMemcpyHostToDevice);

      // Down Sweep
      timer().startGpuTimer();
      for (int d = logN - 1; d >= 0; --d)
      {
        const int powDP1 = std::pow(2, d + 1);
        kernel_DownSweep<<<fullBlocksPerGrid, blockSize>>>(numObjects, powDP1, loopInputBuffer);
      }

      // 3. Store in OData
      Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(numObjects, device_odata, device_idata, device_bools, device_scannedBools);
      timer().endGpuTimer();

      int boolArrayLast = 0;
      cudaMemcpy(&boolArrayLast, &device_bools[nearestPower2 - 1], sizeof(int), cudaMemcpyDeviceToHost);

      int scannedLast = 0;
      cudaMemcpy(&scannedLast, &device_scannedBools[nearestPower2 - 1], sizeof(int), cudaMemcpyDeviceToHost);

      // Eg:
      // 0101 is our bools
      // Scanned: 0 0 1 1 (Exclusive)
      //
      // 01010 is our bools
      // Scanned: 0 0 1 1 2
      //
      // So we add bools[last] + Scanned[last] to get final count.
      // In 1st case: 1 + 1 = 2 entries (final compaction count)
      // In 2nd case: 2 + 0 = 2 entries (final compaction count)
      const int totalEntries = scannedLast + boolArrayLast;
      cudaMemcpy(odata, device_odata, sizeof(int) * (totalEntries), cudaMemcpyDeviceToHost);

      cudaFree(device_idata);
      cudaFree(device_odata);
      cudaFree(device_bools);
      cudaFree(device_scannedBools);
      return totalEntries;
    }
  }
}
