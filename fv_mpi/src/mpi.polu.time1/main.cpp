#include <iostream>
using std::cout;
using std::endl;

#include <tk/stopwatch.hpp>

#define PROFILE_COUNTER_CLASS tk::Stopwatch
#define PROFILE_COUNTER_NAME  s
#define PROFILE_COUNTER       profile::PROFILE_COUNTER_NAME

namespace profile {
	PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;

	long long mltotus;

	void init() {
		PROFILE_COUNTER_NAME = new PROFILE_COUNTER_CLASS();
	}

	inline void output(std::ostream& out) {
		mltotus = PROFILE_COUNTER_NAME->last().microseconds();
		out << mltotus << ';' << endl;
	}

	void cleanup() {
		delete PROFILE_COUNTER_NAME;
	}
}

#define PROFILE_INIT() profile::init()
#define PROFILE_RETRIEVE_CF() ;
#define PROFILE_RETRIEVE_UP() ;
#define PROFILE_RETRIEVE_CM() ;
#define PROFILE_OUTPUT() profile::output(cout);
#define PROFILE_CLEANUP()
#define PROFILE
#define PROFILE1
#define PROFILE_START() PROFILE_COUNTER->start();
#define PROFILE_STOP() PROFILE_COUNTER->stop();

#include "../mpi.polu.time/config.h"
#include "../mpi.polu.time/main.cpp"
