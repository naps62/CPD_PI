#include "papi/computational_intensity.hpp"

#define BUS_TRANS_MEM      0x40000114
#define INST_RETIRED_OTHER 0x4000015f

PAPI_ComputationalIntensity::PAPI_ComputationalIntensity () : PAPI_Preset ()
{
	this->add_event( BUS_TRANS_MEM );
	this->add_event( INST_RETIRED_OTHER );
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
