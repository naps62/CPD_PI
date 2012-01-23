#pragma once
#define _H_CPROFILE

#include <cuda.h>
#include <cuda_runtime.h>

#include "MFVLib_config.h"
#include "MFVLog.h"

namespace CudaFV {

	class CProfile {
		private:
		FVLog *stream;

		public:
		cudaEvent_t start_t, stop_t;
		float time;

		CProfile(string msg);
		CProfile(string msg, string filename);
		~CProfile();

		void start();
		void stop();
		float getTime();

		private:
		void init(string msg, string filename);
	};
}
