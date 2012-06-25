/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVProfile.h
 ** Profiling library for cuda events
 **
 ** Author:			Miguel Palhas, mpalhas@gmail.com
 ** Created:		13-02-2012
 ** Last Tested:	---
 ** \todo for now this file does nothing when nvcc is not being used. maybe change that if it makes sense?
 ** \todo documentation for the rest of this file
 ** -------------------------------------------------------------------------*/

#ifndef _H_CFVPROFILE
#define _H_CFVPROFILE

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#include "FVL/FVGlobal.h"
#include "FVL/FVLog.h"

namespace FVL {

	class CFVProfile {
		private:
		string msg;

		public:
		static FVLog stream; //TODO por esta coisa privada
		cudaEvent_t start_t, stop_t;
		float time;

		CFVProfile(string msg);
		CFVProfile(string msg, string filename);
		~CFVProfile();

		void start();
		void stop();
		float getTime();

		private:
		void init(string msg, string filename);
	};
}

#endif // __CUDACC__

#endif // _H_CFVPROFILE
