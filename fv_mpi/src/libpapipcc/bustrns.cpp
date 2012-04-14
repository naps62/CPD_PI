#include <papi/bustrns.hpp>

namespace papi
{
	namespace events
	{
		MemoryBusTransactionsNativeEvent::MemoryBusTransactionsNativeEvent()
		: Event( "BUS_TRANS_MEM" )
		{}
		
		SelfInstructionFetchBusTransactionsNativeEvent::SelfInstructionFetchBusTransactionsNativeEvent()
		: Event( "BUS_TRANS_IFETCH:SELF" )
		{}
	}

	namespace counters
	{
		MemoryBusTransactionsCounter::MemoryBusTransactionsCounter()
		{
			this->add_event( _btm.code() );
		}

		long long int
		MemoryBusTransactionsCounter::transactions()
		{
			return this->last( _btm.code() );
		}



		SelfInstructionFetchBusTransactionsCounter::SelfInstructionFetchBusTransactionsCounter()
		{
			this->add_event( _btsif.code() );
		}

		long long int
		SelfInstructionFetchBusTransactionsCounter::transactions()
		{
			return this->last( _btsif.code() );
		}
	}
}
