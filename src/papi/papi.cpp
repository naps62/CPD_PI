#include "papi.hpp"

void PAPI::init ()
{
	int result;

	result = PAPI_library_init( PAPI_VER_CURRENT );
	if ( result != PAPI_VER_CURRENT )
		throw( result );
	
	/*
	result = PAPI_thread_init( omp_get_thread_num );
	if ( result != PAPI_OK )
		throw( result );
		*/
}

void PAPI::shutdown ()
{
	PAPI_shutdown();
}

PAPI::PAPI()
{
	int result;

	if ( ! PAPI_is_initialized() )
		PAPI::init();
	
	result = PAPI_create_eventset( &set );
	if (result != PAPI_OK)
		throw( result );
	
	_values = NULL;
}

void PAPI::add_event (int event)
{
	int result;
	result = PAPI_add_event( set , event );
	if (result != PAPI_OK)
		throw( result );
	
	events.push_back( event );
	counters[ event ] = 0;
}

void PAPI::add_events (int *events_v, int events_c)
{
	int i;
	int result;

	result = PAPI_add_events( set , events_v , events_c );
	if (result != PAPI_OK)
		throw( result );
	
	for ( i = 0 ; i < events_c ; ++i )
	{
		events.push_back( events_v[i] );
		counters[ events_v[i] ] = 0;
	}
}

void PAPI::start ()
{
	int result;

	_values = new long long int[ events.size() ];

	result = PAPI_start( set );
	if (result != PAPI_OK)
		throw( result );
}

void PAPI::stop ()
{
	int result;
	unsigned i;

	result = PAPI_stop( set , _values );
	if (result != PAPI_OK)
		throw( result );
	
	for ( i = 0 ; i < events.size() ; ++i )
		counters[ events[i] ] = _values[i];
	
	delete _values;
	_values = NULL;
}

void PAPI::reset ()
{
	int result;
	map< int , long long int >::iterator it;

	result = PAPI_reset( set );

	for ( it = counters.begin(); it != counters.end(); ++it )
		it->second = 0;
}

long long int PAPI::operator[] (int event)
{
	return counters[ event ];
}
