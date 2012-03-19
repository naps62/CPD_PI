#ifndef ___PAPI_COUNTER_PRESET_VECINS___
#define ___PAPI_COUNTER_PRESET_VECINS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct VectorInstructionsPresetEvent : public Event
	{
		VectorInstructionsPresetEvent();
	};





	class VectorInstructionsCounter : public PAPI_Custom
	{
		VectorInstructionsPresetEvent _vecins;
	public:
		VectorInstructionsCounter();

		//
		//  getter
		//
		long long int instructions();
	};
}

#endif//___PAPI_COUNTER_PRESET_VECINS___
