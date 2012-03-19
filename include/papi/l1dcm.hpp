#ifndef ___PAPI_COUNTER_PRESET_L1DCM___
#define ___PAPI_COUNTER_PRESET_L1DCM___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct L1DataCacheMissesPresetEvent : public Event
	{
		L1DataCacheMissesPresetEvent();
	};





	class L1DataCacheMissesCounter : public PAPI_Custom
	{
		L1DataCacheMissesPresetEvent _l1dcm;
	public:
		L1DataCacheMissesCounter();

		//
		//  getter
		//
		long long int misses();
	};
}

#endif//___PAPI_COUNTER_PRESET_L1DCM___
