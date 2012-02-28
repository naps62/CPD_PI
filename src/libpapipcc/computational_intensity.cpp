#include "papi/computational_intensity.hpp"

#define BUS_TRANS_MEM        0x40000114
#define INSTRUCTIONS_RETIRED 0x40000001
#define INST_RETIRED_LOADS   0x4000000e
#define INST_RETIRED_STORES  0x4000000f

PAPI_ComputationalIntensity::PAPI_ComputationalIntensity () : PAPI_Preset ()
{
	this->add_event( BUS_TRANS_MEM );
	this->add_event( INSTRUCTIONS_RETIRED );
	this->add_event( INST_RETIRED_LOADS );
	this->add_event( INST_RETIRED_STORES );
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
	return (*this)[ INSTRUCTIONS_RETIRED ] - (*this)[ INST_RETIRED_LOADS ] - (*this)[ INST_RETIRED_STORES ];
}

double PAPI_ComputationalIntensity::intensity ()
{
	return instructions() / bytes_accessed();
}
