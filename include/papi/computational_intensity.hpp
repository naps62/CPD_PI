#ifndef ___PAPI_COMPUTATIONAL_INTENSITY_HPP___
#define ___PAPI_COMPUTATIONAL_INTENSITY_HPP___

#include <papi/papi.hpp>
#include <papi/bus_trans_mem_event.hpp>
#include <papi/inst_retired_other_event.hpp>

class PAPI_ComputationalIntensity : public PAPI_Preset
{
	papi::BusTransMemEvent _btm;
	papi::InstRetiredOtherEvent _iro;
public:
	PAPI_ComputationalIntensity ();

	long long int ram_accesses ();
	long long int bytes_accessed ();
	long long int instructions ();
	double intensity ();
};

#endif/*___PAPI_COMPUTATIONAL_INTENSITY_HPP___*/
