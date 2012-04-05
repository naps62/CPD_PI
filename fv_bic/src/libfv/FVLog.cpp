/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** FVLog.cpp
** Logging
**
** Author: Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#include "FVL/FVLog.h"

namespace FVL {
	FVLog FVLog::log;
	
	FVLog::FVLog() : ofstream(FV_LOGFILE.c_str(), ofstream::out | FV_LOGMODE ) {
		initLog();
	}
	
	FVLog::FVLog(string filename) : ofstream(filename.c_str(), ofstream::out | FV_LOGMODE ) {
		initLog();
	}

	
	string FVLog::timestamp() {
		time_t ltime; /* calendar time */
		ltime=time(NULL); /* get current cal time */
		return string(asctime(localtime(&ltime)));
	}
	
	void FVLog::initLog() {
		*this << endl << " --- Logger started at " << timestamp() << endl;
	}
}

