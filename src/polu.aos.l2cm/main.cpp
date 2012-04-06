#include <iostream>
#include <limits>

#include <papi/papi.hpp>
#include <papi/cache.hpp>

#define PROFILE_LIMITED 1000
#define PROFILE_WARMUP   100



#define PROFILE_COUNTER_CLASS   papi::counters::L2CacheMissesCounter
#define PROFILE_COUNTER_NAME    p
#define PROFILE_COUNTER_FIELD_0 data
#define PROFILE_COUNTER_FIELD_1 instruction
#define PROFILE_COUNTER_FIELD_2 total
#define PROFILE_COUNTER         profile::PROFILE_COUNTER_NAME



namespace profile
{
	PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;
	long long int PROFILE_COUNTER_FIELD_0;
	long long int PROFILE_COUNTER_FIELD_1;
	long long int PROFILE_COUNTER_FIELD_2;

	long long int cftotns;
	long long int cfminns;
	long long int cfmaxns;

	long long int uptotns;
	long long int upminns;
	long long int upmaxns;

	struct Overhead
	{
		long long int PROFILE_COUNTER_FIELD_0;
		long long int PROFILE_COUNTER_FIELD_1;
		long long int PROFILE_COUNTER_FIELD_2;
		long long int nanoseconds;
	} overhead;

	void init()
	{
		papi::init();

		PROFILE_COUNTER_NAME = new PROFILE_COUNTER_CLASS();

		PROFILE_COUNTER_FIELD_0 = 0;
		PROFILE_COUNTER_FIELD_1 = 0;
		PROFILE_COUNTER_FIELD_2 = 0;
		
		PROFILE_COUNTER_NAME->start();
		PROFILE_COUNTER_NAME->stop();
		overhead.PROFILE_COUNTER_FIELD_0 = PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_0();
		overhead.PROFILE_COUNTER_FIELD_1 = PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_1();
		overhead.PROFILE_COUNTER_FIELD_2 = PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_2();
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
			<<	PROFILE_COUNTER_FIELD_0	<<	';'
			<<	PROFILE_COUNTER_FIELD_1	<<	';'
			<<	PROFILE_COUNTER_FIELD_2	<<	';'
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
		PROFILE_COUNTER_FIELD_0 += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_0() - overhead.PROFILE_COUNTER_FIELD_0;
		PROFILE_COUNTER_FIELD_1 += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_1() - overhead.PROFILE_COUNTER_FIELD_1;
		PROFILE_COUNTER_FIELD_2 += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_2() - overhead.PROFILE_COUNTER_FIELD_2;
		long long int timens = PROFILE_COUNTER_NAME->last();
		cftotns += timens;
		cfminns = ( timens < cfminns ) ? timens : cfminns;
		cfmaxns = ( timens > cfmaxns ) ? timens : cfmaxns;
	}

	inline
	void update()
	{
		PROFILE_COUNTER_FIELD_0 += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_0() - overhead.PROFILE_COUNTER_FIELD_0;
		PROFILE_COUNTER_FIELD_1 += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_1() - overhead.PROFILE_COUNTER_FIELD_1;
		PROFILE_COUNTER_FIELD_2 += PROFILE_COUNTER_NAME->PROFILE_COUNTER_FIELD_2() - overhead.PROFILE_COUNTER_FIELD_2;
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

#include "../polu.aos.papi/main.cpp"
