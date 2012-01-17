#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#else
#warning "Compiler does not present OpenMP support"
#endif

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
	
}

#ifdef _OPENMP
void PAPI::init_threads ()
{
	int result;

	if ( ! PAPI_is_initialized() )
		PAPI::init();

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
#endif

void PAPI::shutdown ()
{
	PAPI_shutdown();
}

long long int PAPI::real_nano_seconds ()
{
	return PAPI_get_real_nsec();
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
	time.last = 0;
	time.total = 0;
	time.avg = 0;
	measures = 0;
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

	time._begin = PAPI::real_nano_seconds();

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
	long long int _end;
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

	_end = PAPI::real_nano_seconds();
	
	for ( i = 0 ; i < events.size() ; ++i )
		counters[ events[i] ] = _values[i];
	
	time.last = _end - time._begin;
	time.total += time.last;
	time.avg = (double) time.total / (double) (++measures);

	delete _values;
	_values = NULL;
}

long long int PAPI::last_time ()
{
	return time.last;
}

void PAPI::reset ()
{
	int result;
	map< int , long long int >::iterator it;

	result = PAPI_reset( set );

	for ( it = counters.begin(); it != counters.end(); ++it )
		it->second = 0;
	
	measures = 0;
	time.last = 0;
	time.total = 0;
	time.avg = 0;
}

long long int PAPI::operator[] (int event)
{
	return counters[ event ];
}

//	PAPI_CPI
PAPI_CPI::PAPI_CPI () : PAPI()
{
	(*this).add_event( PAPI_TOT_INS );
	(*this).add_event( PAPI_TOT_CYC );
}

void PAPI_CPI::add_event (int event)
{
	PAPI::add_event( event );
}

void PAPI_CPI::add_events (int *events_v, int events_c)
{
	PAPI::add_events( events_v , events_c );
}

double PAPI_CPI::cpi ()
{
	return (double)(*this)[ PAPI_TOT_CYC ] / (double)(*this)[ PAPI_TOT_INS ];
}

double PAPI_CPI::ipc ()
{
	return (double)(*this)[ PAPI_TOT_INS ] / (double)(*this)[ PAPI_TOT_CYC ];
}
