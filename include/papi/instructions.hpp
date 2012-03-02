#ifndef ___INSTRUCTIONS_HPP___
#define ___INSTRUCTIONS_HPP___

#include <papi/papi.hpp>
#include <papi/inst_retired_other_event.hpp>

namespace papi
{
	class Instructions : public PAPI_Preset
	{
		InstRetiredOtherEvent _iro;
	public:
		//
		//    constructor
		//
		Instructions();



		//
		//    getter
		//
		long long int instructions();
	};
}

#endif//___INSTRUCTIONS_HPP___
