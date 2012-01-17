#ifndef ___PAPI_HPP___
#define ___PAPI_HPP___

#include <vector>

using std::vector;

class PAPI
{
	int set;
	vector<int> events;
	vector<long long int> values;
	long long int* _values;

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
