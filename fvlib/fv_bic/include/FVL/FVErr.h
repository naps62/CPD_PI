/**
 * \file FVErr.h
 *
 * \brief Error logging class for FVL
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVERR
#define _H_FVERR

#include <iostream>
#include <string>
#include <sstream>
using std::stringstream;
using std::cerr;

#include "FVL/FVLog.h"

namespace FVL {

	/**
	 * An Error logging helper for FVL
	 */
	class FVErr {
		private:
			static FVLog err_log;
	
		public:
			
			/**
			 * Inserts an error message in the error log file
			 *
			 * \param msg Message to append
			 * \param err_code a code to append to the message
			 */
			static void error(string &msg, int err_code);
	
			/**
			 * Inserts a warning message in the error log file
			 *
			 * \param msg Message to append
			 * \param err_code a code to append to the message
			 */
			static void warn(string &msg);
	
		private:
			static void output(FV_LogType type, string &msg);
	};
}

#endif // _H_FVERR

