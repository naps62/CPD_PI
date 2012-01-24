#include "CUDA/CFVProfile.h"

namespace CudaFV {

	FVLog CFVProfile::stream(DEF_PROFILE);

	CFVProfile::CFVProfile(string msg) {
		init(msg, DEF_PROFILE);
	}

	CFVProfile::~CFVProfile() {
		cudaEventDestroy(start_t);
		cudaEventDestroy(stop_t);
	}

	void CFVProfile::start() {
		cudaEventRecord(start_t, 0);
	}

	void CFVProfile::stop() {
		cudaEventRecord(stop_t, 0);
		cudaEventSynchronize(stop_t);
		cudaEventElapsedTime(&time, start_t, stop_t);
		stream << "EVENT: " << msg << ": ";
		stream << time << endl;
	}

	float CFVProfile::getTime() {
		return time;
	}

	void CFVProfile::init(string msg, string filename) {
		this->msg = msg;
		cudaEventCreate(&start_t);
		cudaEventCreate(&stop_t);
	}
}
