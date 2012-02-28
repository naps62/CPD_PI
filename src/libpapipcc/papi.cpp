#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#else
#warning "Compiler does not present OpenMP support"
#endif

#include <papi/papi.hpp>

#define	max(x,y)	\
	( (x > y) ? x : y )

using std::cerr;
using std::endl;

const int PAPI::CACHE_LINE_SIZE = 64;

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
	
	reset();

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

long long int PAPI::total_time ()
{
	return time.total;
}

void PAPI::reset ()
{
	int result;
	map< int , long long int >::iterator it;

	result = PAPI_reset( set );

	for ( it = counters.begin(); it != counters.end(); ++it )
		it->second = 0;
	
	time.last = 0;
	time.total = 0;
	time.avg = 0;
	measures = 0;
}

long long int PAPI::get (int event)
{
	return counters[ event ];
}

long long int PAPI::operator[] (int event)
{
	return counters[ event ];
}

//	PAPI_Custom
void PAPI_Custom::add_event (int event)
{
	PAPI::add_event( event );
}

void PAPI_Custom::add_events (int *events_v, int events_c)
{
	PAPI::add_events( events_v , events_c );
}

//	PAPI_Preset
PAPI_Preset::PAPI_Preset () : PAPI () {}

PAPI_Preset::PAPI_Preset (int *events_v, int events_c) : PAPI ()
{
	PAPI::add_events( events_v , events_c );
}

//	PAPI_Memory
PAPI_Memory::PAPI_Memory () : PAPI_Preset ()
{
	(*this).add_event( PAPI_TOT_INS );
	(*this).add_event( PAPI_LD_INS );
	(*this).add_event( PAPI_SR_INS );
}

long long int PAPI_Memory::loads ()
{
	return (*this)[ PAPI_LD_INS ];
}

long long int PAPI_Memory::stores ()
{
	return (*this)[ PAPI_SR_INS ];
}

//	PAPI_CPI
PAPI_CPI::PAPI_CPI () : PAPI_Preset ()
{
	(*this).add_event( PAPI_TOT_INS );
	(*this).add_event( PAPI_TOT_CYC );
}

long long int PAPI_CPI::cycles ()
{
	return (*this)[ PAPI_TOT_CYC ];
}

long long int PAPI_CPI::instructions ()
{
	return (*this)[ PAPI_TOT_INS ];
}

double PAPI_CPI::cpi ()
{
	return (double)(*this)[ PAPI_TOT_CYC ] / (double)(*this)[ PAPI_TOT_INS ];
}

double PAPI_CPI::ipc ()
{
	return (double)(*this)[ PAPI_TOT_INS ] / (double)(*this)[ PAPI_TOT_CYC ];
}

//	PAPI_Flops
PAPI_Flops::PAPI_Flops () : PAPI_Preset ()
{
	(*this).add_event( PAPI_TOT_CYC );
	(*this).add_event( PAPI_FP_OPS );
}

long long int PAPI_Flops::cycles ()
{
	return (*this)[ PAPI_TOT_CYC ];
}

long long int PAPI_Flops::flops ()
{
	return (*this)[ PAPI_FP_OPS ];
}

double PAPI_Flops::flops_per_cyc ()
{
	return (double)(*this)[ PAPI_FP_OPS ] / (double)(*this)[ PAPI_TOT_CYC ];
}

double PAPI_Flops::flops_per_sec ()
{
	return (double)(*this)[ PAPI_FP_OPS ] * 10e9 / (double)total_time();
}

//	PAPI_Cache
PAPI_Cache::PAPI_Cache () : PAPI_Preset () {}

//	PAPI_L1
PAPI_L1::PAPI_L1 () : PAPI_Cache ()
{
	(*this).add_event( PAPI_L1_DCA );
	(*this).add_event( PAPI_L1_DCM );
}

long long int PAPI_L1::accesses ()
{
	return (*this)[ PAPI_L1_DCA ];
}

long long int PAPI_L1::misses ()
{
	return (*this)[ PAPI_L1_DCM ];
}

double PAPI_L1::miss_rate ()
{
	return (double) misses() / (double) accesses();
}

//	PAPI_L2
PAPI_L2::PAPI_L2 () : PAPI_Cache ()
{
	(*this).add_event( PAPI_L2_DCA );
	(*this).add_event( PAPI_L2_DCM );
}

long long int PAPI_L2::accesses ()
{
	return (*this)[ PAPI_L2_DCA ];
}

long long int PAPI_L2::misses ()
{
	return (*this)[ PAPI_L2_DCM ];
}

double PAPI_L2::miss_rate ()
{
	return (double) misses() / (double) accesses();
}

//	PAPI_OpIntensity
/*
PAPI_OpIntensity::PAPI_OpIntensity () : PAPI_Preset ()
{
	(*this).add_event( PAPI_TOT_CYC );
	(*this).add_event( PAPI_FP_OPS );
	(*this).add_event( PAPI_L2_DCM );
}

long long int PAPI_OpIntensity::ram_accesses ()
{
	return (*this)[ PAPI_L2_DCM ];
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
*/

//	PAPI_InstPerByte
PAPI_InstPerByte::PAPI_InstPerByte ()
{
	(*this).add_event( PAPI_TOT_INS );
	(*this).add_event( PAPI_L2_DCM );
}

long long int PAPI_InstPerByte::instructions ()
{
	return (*this)[ PAPI_TOT_INS ];
}

long long int PAPI_InstPerByte::ram_accesses ()
{
	return (*this)[ PAPI_L2_DCM ];
}

long long int PAPI_InstPerByte::bytes_accessed ()
{
	return (*this)[ PAPI_L2_DCM ] * PAPI::CACHE_LINE_SIZE;
}

double PAPI_InstPerByte::inst_per_byte ()
{
	return instructions() / max( bytes_accessed() , 1 );
}

//	PAPI_MulAdd
PAPI_MulAdd::PAPI_MulAdd ()
{
	(*this).add_event( PAPI_FP_INS );
	(*this).add_event( PAPI_FML_INS );
	(*this).add_event( PAPI_FDV_INS );
}

long long int PAPI_MulAdd::mults ()
{
	return (*this)[ PAPI_FML_INS ];
}

long long int PAPI_MulAdd::divs ()
{
	return (*this)[ PAPI_FDV_INS ];
}

long long int PAPI_MulAdd::adds ()
{
	return (*this)[ PAPI_FP_INS ] - mults() - divs();
}

double PAPI_MulAdd::balance ()
{
	long long int a;
	long long int m;

	a = adds();
	m = mults();

	return (a > m) ? (double) m / (double) a : (double) a / (double) m;
}

//	PAPI_Stopwatch
PAPI_Stopwatch::PAPI_Stopwatch() : _running(false) , _total(0)
{
	if ( ! PAPI_is_initialized() )
		PAPI::init();
}

void PAPI_Stopwatch::start()
{
	if ( ! _running )
	{
		_partial = 0;
		_running = ! _running;
		_begin = PAPI::real_nano_seconds();
	}
}

void PAPI_Stopwatch::stop()
{
	if ( _running )
	{
		_end = PAPI::real_nano_seconds();
		_running = ! _running;
		_partial = _end - _begin;
		_total += _partial;
	}
}

void PAPI_Stopwatch::reset()
{
	if ( ! _running )
		_total = 0;
}
