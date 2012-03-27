#ifndef ___PAPI_INSTRUCTIONS_HPP___
#define ___PAPI_INSTRUCTIONS_HPP___

#include <papi/counter.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct InstructionsPresetEvent
	: public PresetEvent
	{
	protected:
		InstructionsPresetEvent(string name);
	};

	struct BranchInstructionsPresetEvent
	: public InstructionsPresetEvent
	{
		BranchInstructionsPresetEvent();
	};

	struct FloatingPointInstructionsPresetEvent
	: public InstructionsPresetEvent
	{
		FloatingPointInstructionsPresetEvent();
	};

	struct LoadInstructionsPresetEvent
	: public InstructionsPresetEvent
	{
		LoadInstructionsPresetEvent();
	};

	struct StoreInstructionsPresetEvent
	: public InstructionsPresetEvent
	{
		StoreInstructionsPresetEvent();
	};

	struct TotalInstructionsPresetEvent
	: public InstructionsPresetEvent
	{
		TotalInstructionsPresetEvent();
	};

	struct VectorInstructionsPresetEvent
	: public InstructionsPresetEvent
	{
		VectorInstructionsPresetEvent();
	};






	struct InstructionsCounter
	: public Counter
	{
		//
		//  getter
		//

		/// Retrieves the last measured number of instructions.
		virtual long long int instructions_l() = 0;

		/// Retrieves the total measured number of instructions, since creation or last reset.
		virtual long long int instructions_t() = 0;
	};

	class BranchInstructionsCounter
	: public InstructionsCounter
	{
		BranchInstructionsPresetEvent _brins;
	public:
		BranchInstructionsCounter();

		//
		//  getter
		//

		/// Retrieves the last measured value for branch instructions.
		long long int instructions_l();

		/// Retrieves the total measured value for branch instructions, since creation or last reset.
		long long int instructions_t();
	};

	class FloatingPointInstructionsCounter
	: public InstructionsCounter
	{
		FloatingPointInstructionsPresetEvent _fpins;
	public:
		FloatingPointInstructionsCounter();

		//
		//  getter
		//
		
		/// Retrieves the last measured value for floating point instructions.
		long long int instructions_l();

		/// Retrieves the total measured value for floating point instructions, since creation or last reset.
		long long int instructions_t();
	};

	class LoadInstructionsCounter
	: public InstructionsCounter
	{
		LoadInstructionsPresetEvent _ldins;
	public:
		LoadInstructionsCounter();

		//
		//  getter
		//

		/// Retrieves the last measured value for load instructions.
		long long int instructions_l();

		/// Retrieves the total measured value for load instructions, since creation or last reset.
		long long int instructions_t();
	};

	class StoreInstructionsCounter
	: public InstructionsCounter
	{
		StoreInstructionsPresetEvent _srins;
	public:
		StoreInstructionsCounter();

		//
		//  getter
		//

		/// Retrieves the last measured value for store instructions.
		long long int instructions_l();

		/// Retrieves the total measured value for store instructions, since creation or last reset.
		long long int instructions_t();
	};

	class TotalInstructionsCounter
	: public InstructionsCounter
	{
		TotalInstructionsPresetEvent _totins;
	public:
		TotalInstructionsCounter();

		//
		//  getter
		//

		/// Retrieves the last measured value for the total completed instructions.
		long long int instructions_l();

		/// Retrieves the total measured value for the total completed instructions, since creation or last reset.
		long long int instructions_t();
	};

	class VectorInstructionsCounter
	: public InstructionsCounter
	{
		VectorInstructionsPresetEvent _vecins;
	public:
		VectorInstructionsCounter();

		//
		//  getter
		//

		/// Retrieves the last measured value for the vector instrutions.
		long long int instructions_l();

		/// Retrieves the total measured value for the vector instructions, since creation or last reset.
		long long int instructions_t();
	};
}

#endif//___PAPI_INSTRUCTIONS_HPP___
