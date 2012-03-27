#pragma once
#ifndef ___PAPI_BUS_TRANSACTIONS_HPP___
#define ___PAPI_BUS_TRANSACTIONS_HPP___

#include <papi/event.hpp>

#include <string>
using std::string;

namespace papi
{
	/// Super class for any native event regarding any bus transactions.
	/** This class holds no value other than hierarchy logic.
	 */
	struct BusTransactionsNativeEvent
	: public NativeEvent
	{
	protected:
		BusTransactionsNativeEvent(string name);
	};

	/// Native event for bus transactions with memory.
	struct MemoryBusTransactionsNativeEvent
	: public BusTransactionsNativeEvent
	{
		MemoryBusTransactionsNativeEvent();
	};

	/** This class holds no value other than hierarchy logic.
	 */
	struct InstructionFetchBusTransactionsNativeEvent
	: public BusTransactionsNativeEvent
	{
	protected:
		InstructionFetchBusTransactionsNativeEvent(string name);
	};

	/// Native event for bus transactions regarding the fetch of instructions, only for the current core.
	struct SelfInstructionFetchBusTransactionsNativeEvent
	: public InstructionFetchBusTransactionsNativeEvent
	{
		SelfInstructionFetchBusTransactionsNativeEvent();
	};
}

#endif//___PAPI_BUS_TRANSACTIONS_HPP___
