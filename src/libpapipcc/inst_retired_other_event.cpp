#include <papi/inst_retired_other_event.hpp>

namespace papi
{
	InstRetiredOtherEvent::InstRetiredOtherEvent()
		: Event( "INST_RETIRED:OTHER" )
	{}
}
