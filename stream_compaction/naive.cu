#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <valarray>

#define blockSize 256

namespace StreamCompaction
{
  namespace Naive
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

    __global__ void kernel_NaiveParallelScan(int N, int powD, int* odata, int* idata)
    {
      const int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= N) {
        return;
      }

      if (index < powD)
      {
        odata[index] = idata[index];
        return;
      }

      odata[index] = idata[index - powD] + idata[index];
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int* odata, const int* idata)
    {
      numObjects = n;
      cudaMalloc((void**)&device_idata, numObjects * sizeof(int));
      checkCUDAError("cudaMalloc device_idata failed!");

      cudaMalloc((void**)&device_odata, numObjects * sizeof(int));
      checkCUDAError("cudaMalloc device_odata failed!");

      cudaMemcpy(device_idata, idata, sizeof(int) * numObjects, cudaMemcpyHostToDevice);

      dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

      const int logN = ilog2ceil(numObjects);

      int* loopInputBuffer = device_idata;
      int* loopOutputBuffer = device_odata;

      timer().startGpuTimer();
      for (int d = 1; d <= logN; ++d)
      {
        const int powD = std::pow(2, d - 1);
        kernel_NaiveParallelScan<<<fullBlocksPerGrid, blockSize>>>(numObjects - 1, powD, loopOutputBuffer, loopInputBuffer);

        int* temp = loopInputBuffer;
        loopInputBuffer = loopOutputBuffer;
        loopOutputBuffer = temp;
      }
      timer().endGpuTimer();

      cudaMemcpy((odata + 1), loopInputBuffer, sizeof(int) * (numObjects - 1), cudaMemcpyDeviceToHost);

      cudaFree(device_idata);
      cudaFree(device_odata);
    }
  }
}
