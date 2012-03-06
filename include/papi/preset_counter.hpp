#ifndef ___PRESET_COUNTER_HPP___
#define ___PRESET_COUNTER_HPP___

#include <papi/counter.hpp>

namespace papi
{
	class PresetCounter : public PAPI
	{
	protected:
		PresetCounter();
		PresetCounter(int *events_v, int events_c);
	};
}

#endif//___PRESET_COUNTER_HPP___
