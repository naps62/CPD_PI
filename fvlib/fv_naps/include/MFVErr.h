/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** MFVErr.h
** Error handling and logging
**
** Author: Miguel Palhas, mpalhas@gmail.com
** -------------------------------------------------------------------------*/

#ifndef _M_FVERR
#define _M_FVERR

#include <iostream>
#include <string>
#include <sstream>
using std::stringstream;
using std::cerr;

#include "MFVLog.h"

class FVErr {
	private:
		static FVLog log;

	public:
		typedef enum _e_MSG_TYPE {
			ERROR,
			WARNING
		} MSG_TYPE;

		static void error(string &msg, int err_code);

		static void warn(string &msg);

	private:
		static void output(MSG_TYPE type, string &msg);
};

#endif // _M_FV
