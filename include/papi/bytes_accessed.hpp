#pragma once
#ifndef ___BYTES_ACCESSED_HPP___
#define ___BYTES_ACCESSED_HPP___

#include <papi/papi.hpp>
#include <papi/bus_trans_mem_event.hpp>
#include <papi/bus_trans_ifetch_self_event.hpp>

namespace papi
{
	class BytesAccessed : public PAPI_Preset
	{
		BusTransMemEvent _btm;
		BusTransIFetchSelfEvent _btif;
	public:
		BytesAccessed ();

		long long int accesses();
		long long int bytes();
	};
}

#endif//___BYTES_ACCESSED_HPP___
