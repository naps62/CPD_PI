#include <papi/bustrns.hpp>

namespace papi
{
	BusTransactionsNativeEvent::BusTransactionsNativeEvent(string name)
	: NativeEvent(name)
	{}



	MemoryBusTransactionsNativeEvent::MemoryBusTransactionsNativeEvent()
	: BusTransactionsNativeEvent( "BUS_TRANS_MEM" )
	{}



	InstructionFetchBusTransactionsNativeEvent::InstructionFetchBusTransactionsNativeEvent(string name)
	: BusTransactionsNativeEvent(name)
	{}



	SelfInstructionFetchBusTransactionsNativeEvent::SelfInstructionFetchBusTransactionsNativeEvent()
	: InstructionFetchBusTransactionsNativeEvent( "BUS_TRANS_IFETCH:SELF" )
	{}
}
