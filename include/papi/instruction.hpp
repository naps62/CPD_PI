#ifndef ___PAPI_INSTRUCTIONS_HPP___
#define ___PAPI_INSTRUCTIONS_HPP___

#include <papi/counter.hpp>
#include <papi/event.hpp>

namespace papi
{
	namespace events
	{
		struct BranchInstructionsPresetEvent
		: public Event
		{
			BranchInstructionsPresetEvent();
		};

		struct FloatingPointInstructionsPresetEvent
		: public Event
		{
			FloatingPointInstructionsPresetEvent();
		};

		struct LoadInstructionsPresetEvent
		: public Event
		{
			LoadInstructionsPresetEvent();
		};

		struct StoreInstructionsPresetEvent
		: public Event
		{
			StoreInstructionsPresetEvent();
		};

		struct TotalInstructionsPresetEvent
		: public Event
		{
			TotalInstructionsPresetEvent();
		};

		struct VectorInstructionsPresetEvent
		: public Event
		{
			VectorInstructionsPresetEvent();
		};
	}

	namespace counters
	{
		struct InstructionsCounter
		: public Counter
		{
			//
			//  getter
			//

			/// Retrieves the last measured number of instructions.
			virtual long long int instructions() = 0;
		};

		class BranchInstructionsCounter
		: public InstructionsCounter
		{
			events::BranchInstructionsPresetEvent _brins;
		public:
			BranchInstructionsCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for branch instructions.
			long long int instructions();
		};

		class FloatingPointInstructionsCounter
		: public InstructionsCounter
		{
			events::FloatingPointInstructionsPresetEvent _fpins;
		public:
			FloatingPointInstructionsCounter();

			//
			//  getter
			//
			
			/// Retrieves the last measured value for floating point instructions.
			long long int instructions();
		};

		class LoadInstructionsCounter
		: public InstructionsCounter
		{
			events::LoadInstructionsPresetEvent _ldins;
		public:
			LoadInstructionsCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for load instructions.
			long long int instructions();
		};

		class StoreInstructionsCounter
		: public InstructionsCounter
		{
			events::StoreInstructionsPresetEvent _srins;
		public:
			StoreInstructionsCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for store instructions.
			long long int instructions();
		};

		class TotalInstructionsCounter
		: public InstructionsCounter
		{
			events::TotalInstructionsPresetEvent _totins;
		public:
			TotalInstructionsCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for the total completed instructions.
			long long int instructions();
		};

		class VectorInstructionsCounter
		: public InstructionsCounter
		{
			events::VectorInstructionsPresetEvent _vecins;
		public:
			VectorInstructionsCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for the vector instrutions.
			long long int instructions();
		};
	}
}

#endif//___PAPI_INSTRUCTIONS_HPP___
