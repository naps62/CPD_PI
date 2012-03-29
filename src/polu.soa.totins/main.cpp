#include <iostream>
#include <limits>
#include <papi/papi.hpp>
#include <papi/instruction.hpp>


#define PROFILE_COUNTER_CLASS    TotalInstructionsCounter
#define PROFILE_COUNTER_FIELD    instructions
#define PROFILE_COUNTER          profile::PROFILE_COUNTER_NAME
#define PROFILE_COUNTER_NAME     p
#define PROFILE_VAR              totins


#define PROFILE_LIMITED          1000
#define PROFILE_WARMUP           100


namespace profile
{
	papi::counters::PROFILE_COUNTER_CLASS PROFILE_COUNTER_NAME;
	long long int PROFILE_VAR;
	struct Overhead
	{
		long long int PROFILE_COUNTER_FIELD;
		long long int nanoseconds;
	} overhead;

	long long int cftotns;
	long long int cfminns;
	long long int cfmaxns;

	long long int uptotns;
	long long int upminns;
	long long int upmaxns;

	void init()
	{
		PROFILE_VAR += 0;
		PROFILE_COUNTER_NAME.start();
		PROFILE_COUNTER_NAME.stop();
		overhead.PROFILE_COUNTER_FIELD = PROFILE_COUNTER_NAME.PROFILE_COUNTER_FIELD();
		overhead.nanoseconds = PROFILE_COUNTER_NAME.last();

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
			<<	PROFILE_VAR	<<	';'
			<<	cftotns	<<	';'
			<<	cfminns	<<	';'
			<<	cfmaxns	<<	';'
			<<	uptotns	<<	';'
			<<	upminns	<<	';'
			<<	upmaxns	<<	';'
								<<	std::endl;
			;
	}
}

namespace profile
{
	/// Retrieves the values measured in the compute_flux function.
	inline
	void compute_flux()
	{
		PROFILE_VAR += PROFILE_COUNTER_NAME.PROFILE_COUNTER_FIELD() - overhead.PROFILE_COUNTER_FIELD;
		long long int timens = PROFILE_COUNTER_NAME.last() - overhead.nanoseconds;
		cftotns += timens;
		cfminns = ( timens < cfminns ) ? timens : cfminns;
		cfmaxns = ( timens > cfmaxns ) ? timens : cfmaxns;
	}

	/// Retrieves the values measured in the update function.
	inline
	void update()
	{
		PROFILE_VAR += PROFILE_COUNTER_NAME.PROFILE_COUNTER_FIELD() - overhead.PROFILE_COUNTER_FIELD;
		long long int timens = PROFILE_COUNTER_NAME.last() - overhead.nanoseconds;
		uptotns += timens;
		upminns = ( timens < upminns ) ? timens : upminns;
		upmaxns = ( timens > upmaxns ) ? timens : upmaxns;
	}
}


#define PROFILE_INIT() profile::init()

#define PROFILE_OUTPUT() profile::output(std::cout)

#define PROFILE_RETRIEVE_CF() profile::compute_flux()

#define PROFILE_RETRIEVE_UP() profile::update()

#define PROFILE_CLEANUP() ;

#define PROFILE


#include "../polu.soa.papi/main.cpp"
