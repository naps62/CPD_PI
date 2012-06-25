/**
 * \file FVLog.h
 *
 * \brief Log class for FVL
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

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

	/**
	 * A Logger helper for FVL
	 */
	class FVLog : public ofstream {
		private:
			static FVLog log;

		public:
	
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/

			/**
			 * Default constructor
			 *
			 * Opens the default log file in the default write mode
			 * Defaults are specified in FVMacros.h
			 */
			FVLog();

			/**
			 * Construtor to create a log file with a custom name
			 *
			 * \param filename File path for the log file to create/append
			 */
			FVLog(string filename);
	
		private:
			/************************************************
			 * PRIVATE METHODS
			 ***********************************************/

			/**
			 * Gives a string with current timestamp
			 *
			 * \return Current timestamp value, as a string
			 */
			string timestamp();
	
			/**
			 * Dumps initial logging message to file
			 */
			void initLog();
	};
}

#endif // _H_FVLOG
