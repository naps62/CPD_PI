#pragma once
#define _H_CPROFILE

#include "MFVLib_config.h"
#include "MFVLog.h"

namespace CudaFV {

	class CProfile {
		private:
		FVLog *stream;

		public:
		cudaEvent_t start_t, stop_t;
		float time;

		CProfile();
		CProfile(string filename);
		~Cprofile();

		void start();
		void stop();
		float getTime();

		private:
		void init(string msg, string filename);
	}
}
