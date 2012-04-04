#include <papi/memacs.hpp>

namespace papi
{
	namespace counters
	{
		TotalMemoryAccessesCounter::TotalMemoryAccessesCounter()
		{
			this->add_event( _btm.code() );
		}

		long long int
		TotalMemoryAccessesCounter::accesses_l()
		{
			return this->last( _btm.code() );
		}

		long long int
		TotalMemoryAccessesCounter::accesses_t()
		{
			return this->total( _btm.code() );
		}



		DataMemoryAccessesCounter::DataMemoryAccessesCounter()
		{
			this->add_event( _btm.code() );
			this->add_event( _btif.code() );
		}

		long long int
		DataMemoryAccessesCounter::accesses_l()
		{
			return this->last( _btm.code() ) - this->last( _btif.code() );
		}

		long long int
		DataMemoryAccessesCounter::accesses_t()
		{
			return this->total( _btm.code() ) - this->total( _btif.code() );
		}
	}
}
