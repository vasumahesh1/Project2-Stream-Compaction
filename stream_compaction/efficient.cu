#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction
{
  namespace Efficient
  {
    int* device_idata;
    int* device_odata;
    int numObjects;

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

      cudaMemcpy(device_idata, idata, sizeof(int) * nearestPower2, cudaMemcpyHostToDevice);

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
      timer().startGpuTimer();
      // TODO
      timer().endGpuTimer();
      return -1;
    }
  }
}
