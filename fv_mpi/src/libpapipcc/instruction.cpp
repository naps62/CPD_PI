#include <papi/instruction.hpp>

namespace papi
{
	namespace events
	{
		BranchInstructionsPresetEvent::BranchInstructionsPresetEvent()
		: Event( "PAPI_BR_INS" )
		{}

		FloatingPointInstructionsPresetEvent::FloatingPointInstructionsPresetEvent()
		: Event( "PAPI_FP_INS" )
		{}

		LoadInstructionsPresetEvent::LoadInstructionsPresetEvent()
		: Event( "PAPI_LD_INS" )
		{}

		StoreInstructionsPresetEvent::StoreInstructionsPresetEvent()
		: Event( "PAPI_SR_INS" )
		{}

		TotalInstructionsPresetEvent::TotalInstructionsPresetEvent()
		: Event( "PAPI_TOT_INS" )
		{}

		VectorInstructionsPresetEvent::VectorInstructionsPresetEvent()
		: Event( "PAPI_VEC_INS" )
		{}
	}

	namespace counters
	{
		BranchInstructionsCounter::BranchInstructionsCounter()
		{
			this->add_event( _brins.code() );
		}

		long long int
		BranchInstructionsCounter::instructions()
		{
			return this->last( _brins.code() );
		}

		FloatingPointInstructionsCounter::FloatingPointInstructionsCounter()
		{
			this->add_event( _fpins.code() );
		}

		long long int
		FloatingPointInstructionsCounter::instructions()
		{
			return this->last( _fpins.code() );
		}

		LoadInstructionsCounter::LoadInstructionsCounter()
		{
			this->add_event( _ldins.code() );
		}

		long long int
		LoadInstructionsCounter::instructions()
		{
			return this->last( _ldins.code() );
		}

		StoreInstructionsCounter::StoreInstructionsCounter()
		{
			this->add_event( _srins.code() );
		}

		long long int
		StoreInstructionsCounter::instructions()
		{
			return this->last( _srins.code() );
		}

		TotalInstructionsCounter::TotalInstructionsCounter()
		{
			this->add_event( _totins.code() );
		}

		long long int
		TotalInstructionsCounter::instructions()
		{
			return this->last( _totins.code() );
		}

		VectorInstructionsCounter::VectorInstructionsCounter()
		{
			this->add_event( _vecins.code() );
		}
		
		long long int
		VectorInstructionsCounter::instructions()
		{
			return this->last( _vecins.code() );
		}
	}
}
