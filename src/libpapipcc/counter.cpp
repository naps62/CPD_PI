#include <papi/counter.hpp>

#include <iostream>
using std::cerr;
using std::endl;

#include <papi/papi.hpp>

#define NOW() PAPI_get_real_nsec()

namespace papi
{
	Counter::Counter()
	{
		if ( ! PAPI_is_initialized() )
			papi::init();

		_set = PAPI_NULL;
		int result = PAPI_create_eventset( &_set );
		if ( result != PAPI_OK )
		{
			cerr
				<<	'['
				<<	result
				<<	"] Error creating event set!"
				<<	endl
				;
			throw( result );
		}

		this->reset();

		_values = NULL;
	}

	void
	Counter::add_event (int event)
	{
		int result = PAPI_add_event( _set , event );
		if ( result != PAPI_OK )
		{
			cerr
				<<	'['
				<<	result
				<<	"] Error adding event!"
				<<	endl
				;
			throw( result );
		}

		_events.push_back( event );
		_lasts[ event ] = 0;
		_totals[ event ] = 0;
	}



	void
	Counter::add_event (int *events_v, unsigned events_c)
	{
		int result = PAPI_add_events( _set , events_v , events_c );
		if ( result != PAPI_OK )
		{
			cerr
				<<	'['
				<<	result
				<<	"] Error adding events!"
				<<	endl
				;
			throw( result );
		}

		for ( unsigned i = 0 ; i < events_c ; ++i )
		{
			_events.push_back( events_v[ i ]  );
			_lasts[ events_v[i] ] = 0;
			_totals[ events_v[i] ] = 0;
		}
	}

	void
	Counter::start()
	{
		_values = new long long int[ _events.size() ];
		Stopwatch::start();
		int result = PAPI_start( _set );
		if ( result != PAPI_OK )
		{
			cerr
				<<	'['
				<<	result
				<<	"] Error starting measure!"
				<<	endl
				;
			throw( result );
		}
	}



	void
	Counter::stop()
	{
		int result = PAPI_stop( _set , _values );
		if ( result != PAPI_OK )
		{
			cerr
				<<	'['
				<<	result
				<<	"] Error stopping measure!"
				<<	endl
				;
			throw( result );
		}
		Stopwatch::stop();
		unsigned size = _totals.size();
		for ( unsigned i = 0 ; i < size ; ++i )
		{
			_lasts[ i ] = _values[ i ];
			_totals[ i ] += _values[ i ];
		}
		delete _values;
		_values = NULL;
	}

	void
	Counter::reset ()
	{
		PAPI_reset( _set );

		map<int,long long int>::iterator tit = _totals.begin();
		map<int,long long int>::iterator pit = _lasts.begin();
		while ( tit != _totals.end()   &&   pit != _lasts.end() )
		{
			tit->second = 0;
			pit->second = 0;
		}

		Stopwatch::reset();
	}
	

	//
	//  getters
	//
	
	long long int
	Counter::last()
	const
	{
		return Stopwatch::last();
	}

	long long int
	Counter::total()
	const
	{
		return Stopwatch::total();
	}



	long long int
	Counter::last (int event)
	{
		return _lasts[ event ];
	}

	long long int
	Counter::total (int event)
	{
		return _totals[ event ];
	}



	long long int
	Counter::operator[] (int event)
	{
		return _totals[ event ];
	}
}

