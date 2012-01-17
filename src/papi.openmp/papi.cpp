#include <iostream>

#include <omp.h>

#include "papi.hpp"

using std::cerr;
using std::endl;

void PAPI::init ()
{
	int result;

	result = PAPI_library_init( PAPI_VER_CURRENT );
	if ( result != PAPI_VER_CURRENT )
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error initializing PAPI!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"PAPI initialized!"
			<<	endl;
	
	result = PAPI_thread_init( (unsigned long (*)(void)) omp_get_thread_num );
	if ( result != PAPI_OK )
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error initializing PAPI threads!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"PAPI threads initialized!"
			<<	endl;
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
	
	set = PAPI_NULL;
	result = PAPI_create_eventset( &set );
	if (result != PAPI_OK)
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error creating event set!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"Event set created!"
			<<	endl;
	
	_values = NULL;
}

void PAPI::add_event (int event)
{
	int result;
	result = PAPI_add_event( set , event );
	if (result != PAPI_OK)
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error adding event!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"Event added!"
			<<	endl;
	
	events.push_back( event );
	counters[ event ] = 0;
}

void PAPI::add_events (int *events_v, int events_c)
{
	int i;
	int result;

	result = PAPI_add_events( set , events_v , events_c );
	if (result != PAPI_OK)
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error adding events!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"Events added!"
			<<	endl;
	
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
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error starting measure!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"Started measure!"
			<<	endl;
}

void PAPI::stop ()
{
	int result;
	unsigned i;

	result = PAPI_stop( set , _values );
	if (result != PAPI_OK)
	{
		cerr
			<<	'['
			<<	result
			<<	"] Error stopping measure!"
			<<	endl;
		throw( result );
	}
	else
		cerr
			<<	"Stopped measure!"
			<<	endl;
	
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
