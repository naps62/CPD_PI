#include <papi/bytes_accessed.hpp>

namespace papi
{
	//
	//    constructor
	//
	BytesAccessed::BytesAccessed()
	{
		this->add_event( _btm.code() );
		this->add_event( _btif.code() );
	}



	//
	//    getters
	//
	long long int BytesAccessed::accesses()
	{
		return (*this)[ _btm.code() ] - (*this)[ _btif.code() ];
	}

	long long int BytesAccessed::bytes()
	{
		return this->accesses() * PAPI::CACHE_LINE_SIZE;
	}
}
