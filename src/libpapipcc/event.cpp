#include <papi/event.hpp>
#include <papi.h>

#include <cstdlib>
#include <cstring>

namespace papi
{
	//
	//    constructor
	//
	Event::Event(const string name)
	{
		this->name( name );
	}



	//
	//    getters
	//
	int Event::code() const
	{
		return _code;
	}


	//
	//    setters
	//
	void Event::name(const string name)
	{
		char * s = strdup( name.c_str() );
		PAPI_event_name_to_code( s , &_code );
		_name = name;
		free(s);
	}



	NativeEvent::NativeEvent(string name)
	: Event(name)
	{}
}
