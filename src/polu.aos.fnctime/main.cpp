#include <iostream>
#include <limits>

#include <tk/stopwatch-inl.hpp>

#define PROFILE_LIMITED 1000
#define PROFILE_WARMUP   100

#define PROFILE_COUNTER_CLASS tk::Stopwatch
#define PROFILE_COUNTER_NAME  s
#define PROFILE_COUNTER       profile::PROFILE_COUNTER_NAME



namespace profile
{
	PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;

	long long int cftotus;
	long long int cfminus;
	long long int cfmaxus;

	long long int uptotus;
	long long int upminus;
	long long int upmaxus;

	void init()
	{
		PROFILE_COUNTER_NAME = new PROFILE_COUNTER_CLASS();

		cftotus = 0;
		cfminus = std::numeric_limits<long long int>::max();
		cfmaxus = std::numeric_limits<long long int>::min();

		uptotus = 0;
		upminus = std::numeric_limits<long long int>::max();
		upmaxus = std::numeric_limits<long long int>::min();
	}

	void output(std::ostream& out)
	{
		out
			<<	cftotus	<<	';'
			<<	cfminus	<<	';'
			<<	cfmaxus	<<	';'
			<<	uptotus	<<	';'
			<<	upminus	<<	';'
			<<	upmaxus	<<	';'
								<<	std::endl
			;
	}

	void cleanup()
	{
		delete PROFILE_COUNTER_NAME;
	}
}

namespace profile
{
	inline
	void compute_flux()
	{
		long long int timeus = PROFILE_COUNTER_NAME->last().microseconds();
		cftotus += timeus;
		cfminus = ( timeus < cfminus ) ? timeus : cfminus;
		cfmaxus = ( timeus > cfmaxus ) ? timeus : cfmaxus;
	}

	inline
	void update()
	{
		long long int timeus = PROFILE_COUNTER_NAME->last().microseconds();
		uptotus += timeus;
		upminus = ( timeus < upminus ) ? timeus : upminus;
		upmaxus = ( timeus > upmaxus ) ? timeus : upmaxus;
	}
}




#define PROFILE_INIT() profile::init()

#define PROFILE_RETRIEVE_CF() profile::compute_flux()

#define PROFILE_RETRIEVE_UP() profile::update()

#define PROFILE_OUTPUT() profile::output(std::cout)

#define PROFILE_CLEANUP()

#define PROFILE

#include "../polu.aos.papi/main.cpp"
