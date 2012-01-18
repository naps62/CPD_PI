/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** MFVLog.h
** Logging
**
** Author: Miguel Palhas, mpalhas@gmail.com
** -------------------------------------------------------------------------*/

#ifndef _H_M_FVLOG
#define _H_M_FVLOG

#include <ctime>
#include <string>
#include <fstream>
using std::string;
using std::ofstream;
using std::endl;

#define DEF_LOGFILE string("FVLib.log")
#define DEF_ERRFILE string("FVLib.err")

class FVLog : public ofstream {
	public:
		static FVLog logger;

		FVLog();

		FVLog(string filename);

	private:
		string timestamp();

		void initLog();
};

#endif // _H_M_FVLOG

