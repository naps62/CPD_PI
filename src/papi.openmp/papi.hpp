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
	struct {
		long long int _begin;
		long long int last;
		long long int total;
		double avg;
	} time;
	unsigned measures;

	public:
	static void init ();
	static void init_threads();
	static void shutdown ();
	static long long int real_seconds ();
	static long long int real_micro_seconds ();
	static long long int real_nano_seconds ();

	PAPI ();
//	~PAPI ();

	void add_event (int event);
	void add_events (int *events_v, int events_c);

	void start ();
	void stop ();

	long long int last_time ();

	void reset();

	long long int operator[] (int event);
};

#endif/*___PAPI_HPP___*/
