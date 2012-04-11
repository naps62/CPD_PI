#include <iostream>
#include <limits>

#include <papi/papi.hpp>
#include <papi/cycles.hpp>

#define PROFILE_LIMITED 1000
#define PROFILE_WARMUP   100

#define PROFILE_COUNTER_CLASS papi::counters::TotalCyclesCounter
#define PROFILE_COUNTER_NAME  p
#define PROFILE_COUNTER_FIELD cycles
#define PROFILE_COUNTER       profile::PROFILE_COUNTER_NAME



namespace profile
{
	PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;
	long long int PROFILE_COUNTER_FIELD;

	long long int cftotns;
	long long int cfminns;
	long long int cfmaxns;

	long long int uptotns;
	long long int upminns;
	long long int upmaxns;

	struct Overhead
	{
		long long int PROFILE_COUNTER_FIELD;
		long long int nanoseconds;
	} overhead;

	void init()
	{
		papi::init();

		PROFILE_COUNTER_NAME = new PROFILE_COUNTER_CLASS();

		PROFILE_COUNTER_FIELD = 0;
		
		PROFILE_COUNTER_NAME->start();
		PROFILE_COUNTER_NAME->stop();
		overhead.PROFILE_COUNTER_FIELD = PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD();
		overhead.nanoseconds = PROFILE_COUNTER_NAME->last();

		cftotns = 0;
		cfminns = std::numeric_limits<long long int>::max();
		cfmaxns = std::numeric_limits<long long int>::min();

		uptotns = 0;
		upminns = std::numeric_limits<long long int>::max();
		upmaxns = std::numeric_limits<long long int>::min();
	}

	void output(std::ostream& out)
	{
		out
			<<	PROFILE_COUNTER_FIELD	<<	';'
			<<	cftotns	<<	';'
			<<	cfminns	<<	';'
			<<	cfmaxns	<<	';'
			<<	uptotns	<<	';'
			<<	upminns	<<	';'
			<<	upmaxns	<<	';'
								<<	std::endl
			;
	}

	void cleanup()
	{
		delete PROFILE_COUNTER_NAME;
		papi::shutdown();
	}
}

namespace profile
{
	inline
	void compute_flux()
	{
		PROFILE_COUNTER_FIELD += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD() - overhead.PROFILE_COUNTER_FIELD;
		long long int timens = PROFILE_COUNTER_NAME->last();
		cftotns += timens;
		cfminns = ( timens < cfminns ) ? timens : cfminns;
		cfmaxns = ( timens > cfmaxns ) ? timens : cfmaxns;
	}

	inline
	void update()
	{
		PROFILE_COUNTER_FIELD += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD() - overhead.PROFILE_COUNTER_FIELD;
		long long int timens = PROFILE_COUNTER_NAME->last();
		uptotns += timens;
		upminns = ( timens < upminns ) ? timens : upminns;
		upmaxns = ( timens > upmaxns ) ? timens : upmaxns;
	}
}




#define PROFILE_INIT() profile::init()

#define PROFILE_RETRIEVE_CF() profile::compute_flux()

#define PROFILE_RETRIEVE_UP() profile::update()

#define PROFILE_OUTPUT() profile::output(std::cout)

#define PROFILE_CLEANUP()

#define PROFILE

#include "../polu.soa.papi/main.cpp"
