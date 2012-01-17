#include "MFVLog.h"

FVLog FVLog::logger;

FVLog::FVLog() : ofstream(DEF_LOGFILE.c_str(), ofstream::out | ofstream::app ) {
	initLog();
}

FVLog::FVLog(string filename) : ofstream(filename.c_str(), ofstream::out | ofstream::app ) {
	initLog();
}

string FVLog::timestamp() {
	time_t ltime; /* calendar time */
	ltime=time(NULL); /* get current cal time */
	return string(asctime(localtime(&ltime)));
}

void FVLog::initLog() {
	*this << endl << endl << " --- Logger started at " << timestamp() << endl;
}
