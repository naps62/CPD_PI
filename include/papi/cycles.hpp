#ifndef ___PAPI_CYCLES_HPP___
#define ___PAPI_CYCLES_HPP___

#include <papi/counter.hpp>
#include <papi/event.hpp>

namespace papi
{
	namespace events
	{
		struct TotalCyclesPresetEvent
		: public Event
		{
			TotalCyclesPresetEvent();
		};
	}

	namespace counters
	{
		struct CyclesCounter
		: public Counter
		{
			virtual long long int cycles() = 0;
		};

		class TotalCyclesCounter
		: public CyclesCounter
		{
			events::TotalCyclesPresetEvent _totcyc;
		public:
			TotalCyclesCounter();

			long long int cycles();
		};
	}
}

#endif//___PAPI_CYCLES_HPP___
