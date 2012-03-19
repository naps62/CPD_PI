#ifndef ___PAPI_COUNTER_PRESET_BRINS___
#define ___PAPI_COUNTER_PRESET_BRINS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct BranchInstructionsPresetEvent : public Event
	{
		BranchInstructionsPresetEvent();
	};





	class BranchInstructionsCounter : public PAPI_Custom
	{
		BranchInstructionsPresetEvent _brins;
	public:
		BranchInstructionsCounter();

		//
		//  getter
		//
		long long int instructions();
	};
}

#endif//___PAPI_COUNTER_PRESET_BRINS___
