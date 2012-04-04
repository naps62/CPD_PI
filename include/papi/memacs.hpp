#pragma once
#ifndef ___PAPI_MEMORY_ACCESSES_HPP___
#define ___PAPI_MEMORY_ACCESSES_HPP___

#include <papi/counter.hpp>
#include <papi/bustrns.hpp>

namespace papi
{
	namespace counters
	{
		struct MemoryAccessesCounter
		: public Counter
		{
			//
			//  getters
			//

			/// Retrieves the last measured number of memory accesses.
			virtual long long int accesses_l() = 0;

			/// Retrieves the total measured number of memory accesses, since creation or last reset.
			virtual long long int accesses_t() = 0;
		};

		class TotalMemoryAccessesCounter
		: public MemoryAccessesCounter
		{
			events::MemoryBusTransactionsNativeEvent _btm;
		public:
			TotalMemoryAccessesCounter();


			//
			//  getters
			//
			
			long long int accesses_l();

			long long int accesses_t();
		};

		class DataMemoryAccessesCounter
		: public MemoryAccessesCounter
		{
			events::MemoryBusTransactionsNativeEvent _btm;
			events::SelfInstructionFetchBusTransactionsNativeEvent _btif;
		public:
			DataMemoryAccessesCounter();



			//
			//  getters
			//
			
			long long int accesses_l();

			long long int accesses_t();
		};
	}
}

#endif//___PAPI_MEMORY_ACCESSES_HPP___
