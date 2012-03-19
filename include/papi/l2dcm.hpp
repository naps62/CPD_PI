#ifndef ___PAPI_COUNTER_PRESET_L2DCM___
#define ___PAPI_COUNTER_PRESET_L2DCM___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct L2DataCacheMissesPresetEvent : public Event
	{
		L2DataCacheMissesPresetEvent();
	};





	class L2DataCacheMissesCounter : public PAPI_Custom
	{
		L2DataCacheMissesPresetEvent _l2dcm;
	public:
		L2DataCacheMissesCounter();

		//
		//  getter
		//
		long long int misses();
	};
}

#endif//___PAPI_COUNTER_PRESET_L2DCM___
