#pragma once
#ifndef ___BUS_TRANS_MEM_EVENT_HPP___
#define ___BUS_TRANS_MEM_EVENT_HPP___

#include <papi/event.hpp>

namespace papi
{
	struct BusTransMemEvent : public Event
	{
		//
		//    constructor
		//
		BusTransMemEvent();
	};
}

#endif//___BUS_TRANS_MEM_EVENT_HPP___
