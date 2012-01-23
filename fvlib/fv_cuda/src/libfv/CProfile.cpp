#include "CUDA/CProfile.h"

namespace CudaFV {

	FVLog *CProfile::stream(DEF_PROFILE);

	CProfile::CProfile(string msg) {
		init(msg, DEF_PROFILE);
	}

	CProfile::~CProfile() {
		delete stream;
		cudaEventDestroy(start_t);
		cudaEventDestroy(stop_t);
	}

	void CProfile::start() {
		cudaEventRecord(start_t, 0);
	}

	void CProfile::stop() {
		cudaEventRecord(stop_t, 0);
		cudaEventSynchronize(stop_t);
		cudaEventElapsedTime(&time, start_t, stop_t);
		*stream << time << endl;
	}

	float CProfile::getTime() {
		return time;
	}

	void CProfile::init(string msg, string filename) {
		stream = new FVLog(DEF_PROFILE);
		*stream << "EVENT: " << msg << ": ";
		cudaEventCreate(&start_t);
		cudaEventCreate(&stop_t);
	}
}
