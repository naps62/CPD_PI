#pragma once
#ifndef ___BUS_TRANS_IFETCH_SELF_HPP___
#define ___BUS_TRANS_IFETCH_SELF_HPP___

#include <papi/event.hpp>

namespace papi
{
	struct BusTransIFetchSelfEvent : public Event
	{
		//
		//    constructor
		//
		BusTransIFetchSelfEvent();
	};
}

#endif//___BUS_TRANS_IFETCH_SELF_HPP___
