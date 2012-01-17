#ifndef ___PAPI_HPP___
#define ___PAPI_HPP___

#include <map>
#include <vector>

#include <papi.h>

using std::map;
using std::vector;

class PAPI
{
	int set;
	long long int* _values;
	map< int , long long int > counters;
	vector< int > events;

	public:
	static void init ();
	static void shutdown ();

	PAPI ();
//	~PAPI ();

	void add_event (int event);
	void add_events (int *events_v, int events_c);

	void start ();
	void stop ();

	void reset();

	long long int operator[] (int event);
};

#endif/*___PAPI_HPP___*/
