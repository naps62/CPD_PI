/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** FVErr.h
** Error handling and logging
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Test:	---
** -------------------------------------------------------------------------*/

#ifndef _H_FVERR
#define _H_FVERR

#include <iostream>
#include <string>
#include <sstream>
using std::stringstream;
using std::cerr;

#include "FVL/FVLog.h"

namespace FVL {

	class FVErr {
		private:
			static FVLog err_log;
	
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
}

#endif // _H_M_FV

