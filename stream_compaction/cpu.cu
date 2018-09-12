#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <memory>

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          odata[0] = 0;
          for (int idx = 1; idx < n; ++idx)
          {
            odata[idx] = odata[idx - 1] + idata[idx - 1];
          }

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

          int outIdx = 0;
          for (int idx = 0; idx < n; ++idx)
          {
            const int inputData = idata[idx];
            if (inputData != 0)
            {
              odata[outIdx] = inputData;
              ++outIdx;
            }
          }

	        timer().endCpuTimer();
          return outIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
          const std::unique_ptr<int[]> conditionArray = std::make_unique<int[]>(n);
          const std::unique_ptr<int[]> scanArray = std::make_unique<int[]>(n);

	        timer().startCpuTimer();

          for (int idx = 0; idx < n; ++idx)
          {
            conditionArray[idx] = idata[idx] != 0 ? 1 : 0;
          }

          scanArray[0] = 0;
          for (int idx = 1; idx < n; ++idx)
          {
            scanArray[idx] = scanArray[idx - 1] + conditionArray[idx - 1];
          }

          int outIdx = 0;
          for (int idx = 0; idx < n; ++idx)
          {
            const int inputData = idata[idx];
            if (conditionArray[idx] == 1)
            {
              outIdx = scanArray[idx];
              odata[outIdx] = inputData;
            }
          }

	        timer().endCpuTimer();

          return outIdx + 1;
        }
    }
}
