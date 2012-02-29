#include "papi/computational_intensity.hpp"

#define BUS_TRANS_MEM      0x40000114
#define INST_RETIRED_OTHER 0x4000015f

PAPI_ComputationalIntensity::PAPI_ComputationalIntensity () : PAPI_Preset ()
{
	int event;
	PAPI_event_name_to_code( "BUS_TRANS_MEM" , &event );
	this->add_event( event );
	PAPI_event_name_to_code( "INST_RETIRED:OTHER" , &event );
	this->add_event( event );
}

long long int PAPI_ComputationalIntensity::ram_accesses ()
{
	return (*this)[ BUS_TRANS_MEM ];
}

long long int PAPI_ComputationalIntensity::bytes_accessed ()
{
	return ram_accesses() * PAPI::CACHE_LINE_SIZE;
}

long long int PAPI_ComputationalIntensity::instructions ()
{
	return (*this)[ INST_RETIRED_OTHER ];
}

double PAPI_ComputationalIntensity::intensity ()
{
	return instructions() / bytes_accessed();
}
