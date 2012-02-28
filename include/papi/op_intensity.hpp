#ifndef ___PAPI_OP_INTENSITY_HPP___
#define ___PAPI_OP_INTENSITY_HPP___

#include <papi/papi.hpp>

class PAPI_OpIntensity : public PAPI_Preset
{
	public:
		PAPI_OpIntensity ();

		long long int ram_accesses ();
		long long int bytes_accessed ();
		long long int flops ();
		double intensity ();
};

#endif/*___PAPI_OP_INTENSITY_HPP___*/
