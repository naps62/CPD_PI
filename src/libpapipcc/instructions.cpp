#include <papi/instructions.hpp>

namespace papi
{
	//
	//    constructor
	//
	Instructions::Instructions()
	{
		this->add_event( _iro.code() );
	}



	//
	//    getter
	//
	long long int Instructions::instructions()
	{
		return (*this)[ _iro.code() ];
	}
}
