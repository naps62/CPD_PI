/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** MFVLog.h
** Logging
**
** Author: Miguel Palhas, mpalhas@gmail.com
** -------------------------------------------------------------------------*/
#pragma once
#define _H_M_FVLOG

#include <ctime>
#include <string>
#include <fstream>
using std::string;
using std::ofstream;
using std::endl;

#include "MFVLib_config.h"

class FVLog : public ofstream {
	public:
		static FVLog logger;

		FVLog();

		FVLog(string filename);

	private:
		string timestamp();

		void initLog();
};
