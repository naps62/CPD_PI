#ifndef ___PAPI_COUNTER_PRESET_FLOPS___
#define ___PAPI_COUNTER_PRESET_FLOPS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct FloatingPointOperationsPresetEvent : public Event
	{
		FloatingPointOperationsPresetEvent();
	};





	class FloatingPointOperationsCounter : public PAPI_Custom
	{
		FloatingPointOperationsPresetEvent _flops;
	public:
		FloatingPointOperationsCounter();

		//
		//  getter
		//
		long long int operations();
	};
}

#endif//___PAPI_COUNTER_PRESET_FLOPS___
