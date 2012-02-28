#ifndef ___PAPI_COMPUTATIONAL_INTENSITY_HPP___
#define ___PAPI_COMPUTATIONAL_INTENSITY_HPP___

#include <papi/papi.hpp>

class PAPI_ComputationalIntensity : public PAPI_Preset
{
	public:
		PAPI_ComputationalIntensity ();

		long long int ram_accesses ();
		long long int bytes_accessed ();
		long long int instructions ();
		double intensity ();
};

#endif/*___PAPI_COMPUTATIONAL_INTENSITY_HPP___*/
