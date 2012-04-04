#pragma once
#ifndef ___PAPI_BUS_TRANSACTIONS_HPP___
#define ___PAPI_BUS_TRANSACTIONS_HPP___

#include <papi/counter.hpp>
#include <papi/event.hpp>

namespace papi
{
	namespace events
	{
		/// Native event for bus transactions with memory.
		struct MemoryBusTransactionsNativeEvent
		: public Event
		{
			MemoryBusTransactionsNativeEvent();
		};

		/// Native event for bus transactions regarding the fetch of instructions, only for the current core.
		struct SelfInstructionFetchBusTransactionsNativeEvent
		: public Event
		{
			SelfInstructionFetchBusTransactionsNativeEvent();
		};
	}

	namespace counters
	{
		struct BusTransactionsCounter
		: public Counter
		{
			virtual long long int transactions() = 0;
		};

		class MemoryBusTransactionsCounter
		: public BusTransactionsCounter
		{
			events::MemoryBusTransactionsNativeEvent _btm;
		public:
			MemoryBusTransactionsCounter();

			//
			//
			//
			long long int transactions();
		};

		class SelfInstructionFetchBusTransactionsCounter
		: public BusTransactionsCounter
		{
			events::SelfInstructionFetchBusTransactionsNativeEvent _btsif;
		public:
			SelfInstructionFetchBusTransactionsCounter();

			//
			//
			//
			long long int transactions();
		};
	}
}

#endif//___PAPI_BUS_TRANSACTIONS_HPP___
