#include "CProfile.h"

namespace CudaFV {

	CProfile(string msg) {
		init(msg, DEF_PROFILE);
	}

	CProfile(string msg, string filename) {
		init(msg, filename);
	}

	~CProfile() {
		delete stream;
		cudaEventDestroy(start_t);
		cudaEventDestroy(stop_t);
	}

	void start() {
		cudaEventRecord(start_t, 0);
	}

	void stop() {
		cudaEventRecord(stop_t, 0);
		cudaEventSynchronize(stop_t);
		cudaEventElapsedTime(&time, start_t, stop_t);
		*stream << time << endl;
	}

	void getTime() {
		return time;
	}

	void init(string msg, string filename) {
		stream = new FVLog(DEF_PROFILE);
		*stream << "EVENT: " << msg ": ";
		cudaEventCreate(&start_t);
		cudaEventCreate(&stop_t);
	}
}
