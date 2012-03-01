#include "papi/computational_intensity.hpp"

PAPI_ComputationalIntensity::PAPI_ComputationalIntensity ()
	: PAPI_Preset ()
{
	this->add_event( _btm.code() );
	this->add_event( _iro.code() );
}

long long int PAPI_ComputationalIntensity::ram_accesses ()
{
	return (*this)[ _btm.code() ];
}

long long int PAPI_ComputationalIntensity::bytes_accessed ()
{
	return ram_accesses() * PAPI::CACHE_LINE_SIZE;
}

long long int PAPI_ComputationalIntensity::instructions ()
{
	return (*this)[ _iro.code() ];
}

double PAPI_ComputationalIntensity::intensity ()
{
	return (double) instructions() / (double) bytes_accessed();
}
