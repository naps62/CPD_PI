#include "papi/op_intensity.hpp"

#define BUS_TRANS_MEM   0x40000114
#define FP_COMP_OPS_EXE 0x4000001e

PAPI_OpIntensity::PAPI_OpIntensity () : PAPI_Preset ()
{
	this->add_event( BUS_TRANS_MEM );
	this->add_event( FP_COMP_OPS_EXE );
}

long long int PAPI_OpIntensity::ram_accesses ()
{
	return (*this)[ BUS_TRANS_MEM ];
}

long long int PAPI_OpIntensity::bytes_accessed ()
{
	return ram_accesses() * PAPI::CACHE_LINE_SIZE;
}

long long int PAPI_OpIntensity::flops ()
{
	return (*this)[ FP_COMP_OPS_EXE ];
}

double PAPI_OpIntensity::intensity ()
{
	return flops() / bytes_accessed();
}
