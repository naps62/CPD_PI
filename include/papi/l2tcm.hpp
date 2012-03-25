#ifndef ___PAPI_COUNTER_PRESET_L2TCM___
#define ___PAPI_COUNTER_PRESET_L2TCM___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct L2TotalCacheMissesPresetEvent : public Event
	{
		L2TotalCacheMissesPresetEvent();
	};





	class L2TotalCacheMissesCounter : public PAPI_Custom
	{
		L2TotalCacheMissesPresetEvent _l2tcm;
	public:
		L2TotalCacheMissesCounter();

		//
		//  getter
		//
		long long int misses();
	};
}

#endif//___PAPI_COUNTER_PRESET_L2TCM___
