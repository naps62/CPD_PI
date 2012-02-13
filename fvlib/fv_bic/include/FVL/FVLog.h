/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** FVLog.h
** Logging
**
** Author: Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#ifndef _H_FVLOG
#define _H_FVLOG

#include <ctime>
#include <string>
#include <fstream>
using std::string;
using std::ofstream;
using std::endl;

#include "FVL/FVGlobal.h"

namespace FVL {
	class FVLog : public ofstream {
		public:
			static FVLog log;
	
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			FVLog();
			FVLog(string filename);
	
		private:
			/************************************************
			 * PRIVATE METHODS
			 ***********************************************/

			// gen a string with current timestamp
			string timestamp();
	
			// saves log start message
			void initLog();
	};
}

#endif // _H_FVLOG
