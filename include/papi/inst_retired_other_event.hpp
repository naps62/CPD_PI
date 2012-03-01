#pragma once
#ifndef ___INST_RETIRED_OTHER_EVENT_HPP___
#define ___INST_RETIRED_OTHER_EVENT_HPP___

#include <papi/event.hpp>

namespace papi
{
	struct InstRetiredOtherEvent : public Event
	{
		//
		//    constructor
		//
		InstRetiredOtherEvent();
	};
}

#endif//___INST_RETIRED_OTHER_EVENT_HPP___
