#ifndef _M_FVERR
#define _M_FVERR

#include <iostream>
#include <string>

#include "FVLog.h"

class FVerr {
	private:
		FVLog log(DEF_ERRFILE);

	public:
		typedef enum _e_MSG_TYPE {
			ERROR,
			WARNING,
		} MSG_TYPE;

		static void error(string &msg, int err_code) {
			output(ERROR, msg);
			exit(err_code);
		}

		static void warn(string &msg) {
			output(WARNING, msg);
		}

	private:
		static output(MSG_TYPE type, string &full) {
			stringstream full_msg;
			switch (type) {
				case ERROR:
					full_msg << "Error: ";
					break;
				case WARNING:
					full_msg << "Warning: ";
					break;
				default:
					full_msg << "Unkown msg type: ";
					break;
			}

			full_msg << msg << endl << endl;
			string full_msg_str = full_msg.str();
			log  << full_msg_str;
			cerr << full_msg_str;
		}
};

#endif // _M_FV
