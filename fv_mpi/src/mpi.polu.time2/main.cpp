#include <iostream>
using std::cout;
using std::endl;

#include <tk/stopwatch.hpp>

#define PROFILE_COUNTER_CLASS tk::Stopwatch
#define PROFILE_COUNTER_NAME  s
#define PROFILE_COUNTER       profile::PROFILE_COUNTER_NAME

namespace profile {
	PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;

	long long cftotus;
	long long cfminus;
	long long cfmaxus;

	long long uptotus;
	long long upminus;
	long long upmaxus;

	long long cmtotus;
	long long cmminus;
	long long cmmaxus;

	void init() {
		PROFILE_COUNTER_NAME = new PROFILE_COUNTER_CLASS();

		cftotus = 0;
		cfminus = numerical_limits<long long>::max();
		cfmaxus = numerical_limits<long long>::min();

		uptotus = 0;
		upminus = numerical_limits<long long>::max();
		upmaxus = numerical_limits<long long>::min();

		cmtotus = 0;
		cmminus = numerical_limits<long long>::max();
		cmmaxus = numerical_limits<long long>::min();
	}

	inline void output(std::ostream& out) {
		out
			<< cftotus << ';' << cfminus << ';' << cfmaxus << ';'
			<< uptotus << ';' << upminus << ';' << upmaxus << ';'
			<< cmtotus << ';' << cmminus << ';' << cmmaxus << ';'
			<< endl;
	}

	void cleanup() {
		delete PROFILE_COUNTER_NAME;
	}

	inline void compute_flux() {
		long long timeus = PROFILE_COUNTER_NAME->last().microseconds();
		cftotus += timeus;
		cfminus = (timeus < cfminus) ? timeus : cfminus;
		cfmaxus = (timeus > cfmaxus) ? timeus : cfmaxus;
	}

	inline void update() {
		long long timeus = PROFILE_COUNTER_NAME->last().microseconds();
		uptotus += timeus;
		upminus = (timeus < upminus) ? timeus : upminus;
		upmaxus = (timeus > upmaxus) ? timeus : upmaxus;
	}

	inline void communication() {
		long long timeus = PROFILE_COUNTER_NAME->last().microseconds();
		cmtotus += timeus;
		cmminus = (timeus < cmminus) ? timeus : cmminus;
		cmmaxus = (timeus > cmmaxus) ? timeus : cmmaxus;
	}
}

#define PROFILE_INIT() profile::init()
#define PROFILE_RETRIEVE_CF() profile::compute_flux()
#define PROFILE_RETRIEVE_UP() profile::update()
#define PROFILE_RETRIEVE_CM() profile::communication()
#define PROFILE_OUTPUT() profile::output(cout);
#define PROFILE_CLEANUP()
#define PROFILE
#define PROFILE2
#define PROFILE_START() PROFILE_COUNTER->start();
#define PROFILE_STOP() PROFILE_COUNTER->stop();

#include "../mpi.polu.time/config.h"
#include "../mpi.polu.time/main.cpp"
