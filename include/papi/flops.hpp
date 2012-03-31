#ifndef ___PAPI_FLOPS_HPP___
#define ___PAPI_FLOPS_HPP___

#include <papi/event.hpp>
#include <papi/counter.hpp>

namespace papi
{
	namespace events
	{
		struct FloatingPointOperationsPresetEvent
		: public Event
		{
			FloatingPointOperationsPresetEvent();
		};
	}

	namespace counters
	{
		struct OperationsCounter
		: public Counter
		{
			virtual long long int operations() = 0;
		};

		class FloatingPointOperationsCounter
		: public OperationsCounter
		{
			events::FloatingPointOperationsPresetEvent _flops;
		public:
			FloatingPointOperationsCounter();

			//
			//  getter
			//

			long long int operations();
		};
	}
}

#endif//___PAPI_FLOPS_HPP___
