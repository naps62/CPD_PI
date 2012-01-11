#include <stdlib.h>

#include "MFVErr.h"

FVLog FVErr::log(DEF_ERRFILE);

void FVErr::error(string &msg, int err_code) {
	output(ERROR, msg);
	exit(err_code);
}

void FVErr::warn(string &msg) {
	output(WARNING, msg);
}

void FVErr::output(MSG_TYPE type, string &msg) {
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
