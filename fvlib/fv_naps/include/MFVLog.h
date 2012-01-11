#ifndef _M_FVLOG
#define _M_FVLOG

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
		FVLog();

		FVLog(string filename);

	private:
		string timestamp();

		void initLog();
};

#endif // _M_FVLOG
