#include <papi/papiinst.hpp>

namespace papi
{
	InstructionsPresetEvent::InstructionsPresetEvent(string name)
	: PresetEvent(name)
	{}

	BranchInstructionsPresetEvent::BranchInstructionsPresetEvent()
	: InstructionsPresetEvent( "PAPI_BR_INS" )
	{}

	FloatingPointInstructionsPresetEvent::FloatingPointInstructionsPresetEvent()
	: InstructionsPresetEvent( "PAPI_FP_INS" )
	{}

	LoadInstructionsPresetEvent::LoadInstructionsPresetEvent()
	: InstructionsPresetEvent( "PAPI_LD_INS" )
	{}

	StoreInstructionsPresetEvent::StoreInstructionsPresetEvent()
	: InstructionsPresetEvent( "PAPI_SR_INS" )
	{}

	TotalInstructionsPresetEvent::TotalInstructionsPresetEvent()
	: InstructionsPresetEvent( "PAPI_TOT_INS" )
	{}

	VectorInstructionsPresetEvent::VectorInstructionsPresetEvent()
	: InstructionsPresetEvent( "PAPI_VEC_INS" )
	{}



	BranchInstructionsCounter::BranchInstructionsCounter()
	{
		this->add_event( _brins.code() );
	}

	long long int
	BranchInstructionsCounter::instructions_l()
	{
		return this->last( _brins.code() );
	}

	long long int
	BranchInstructionsCounter::instructions_t()
	{
		return this->total( _brins.code() );
	}

	FloatingPointInstructionsCounter::FloatingPointInstructionsCounter()
	{
		this->add_event( _fpins.code() );
	}

	long long int
	FloatingPointInstructionsCounter::instructions_l()
	{
		return this->last( _fpins.code() );
	}

	long long int
	FloatingPointInstructionsCounter::instructions_t()
	{
		return this->total( _fpins.code() );
	}

	LoadInstructionsCounter::LoadInstructionsCounter()
	{
		this->add_event( _ldins.code() );
	}

	long long int
	LoadInstructionsCounter::instructions_l()
	{
		return this->last( _ldins.code() );
	}

	long long int
	LoadInstructionsCounter::instructions_t()
	{
		return this->total( _ldins.code() );
	}

	StoreInstructionsCounter::StoreInstructionsCounter()
	{
		this->add_event( _srins.code() );
	}

	long long int
	StoreInstructionsCounter::instructions_l()
	{
		return this->last( _srins.code() );
	}

	long long int
	StoreInstructionsCounter::instructions_t()
	{
		return this->total( _srins.code() );
	}

	TotalInstructionsCounter::TotalInstructionsCounter()
	{
		this->add_event( _totins.code() );
	}

	long long int
	TotalInstructionsCounter::instructions_l()
	{
		return this->last( _totins.code() );
	}

	long long int
	TotalInstructionsCounter::instructions_t()
	{
		return this->total( _totins.code() );
	}

	VectorInstructionsCounter::VectorInstructionsCounter()
	{
		this->add_event( _vecins.code() );
	}
	
	long long int
	VectorInstructionsCounter::instructions_l()
	{
		return this->last( _vecins.code() );
	}

	long long int
	VectorInstructionsCounter::instructions_t()
	{
		return this->total( _vecins.code() );
	}
}
