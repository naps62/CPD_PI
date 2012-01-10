#ifndef _M_FVLOG
#define _M_FVLOG

#include <ctime>
#include <string>
#include <fstream>
using std::string;
using std::ofstream;
using std::endl;

#define DEF_LOGFILE "FVLib.log"
#define DEF_ERRFILE "FVLib.err"

class FVLog : public ofstream {
	public:
		FVLog(string &filename) : ofstream(filename.c_str(), ofstream::out | ofstream::app ) {
			*this << endl << endl << " --- Logger started at " << timestamp() << endl;
		}

	private:
		string timestamp() {
		    time_t ltime; /* calendar time */
	        ltime=time(NULL); /* get current cal time */
	        return string(asctime(localtime(&ltime)));
		}
};

#endif // _M_FVLOG
