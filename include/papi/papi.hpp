#ifndef ___PAPI_HPP___
#define ___PAPI_HPP___

#include <map>
#include <vector>

#include <papi.h>

using std::map;
using std::vector;


namespace papi
{
	//! Initialize the PAPI library funcionalities.
	/*! Initialize the library with the current version. If an error occurs
	 * throws an exception.
	 *
	 * \throw Error An error occurred while initializing the library. The value
	 * of the exception is the error code returned by the PAPI library.
	 */
	void init();
	void shutdown();
}


class PAPI
{
	int set;
	long long int* _values;
	unsigned measures;
	map< int , long long int > counters;
	vector< int > events;
	struct {
		long long int _begin;
		long long int last;
		long long int total;
		double avg;
	} time;

	protected:
	static const int CACHE_LINE_SIZE;

	virtual void add_event (int event);
	virtual void add_events (int *events_v, int events_c);

	public:
	static void init ();
#ifdef _OPENMP
	static void init_threads();
#endif
	static void shutdown ();
	static long long int real_seconds ();
	static long long int real_micro_seconds ();
	static long long int real_nano_seconds ();

	PAPI ();

	void start ();
	void stop ();

	long long int last_time ();
	long long int total_time ();

	void reset();

	long long int get( int event );

	long long int operator[] (int event);
};

class PAPI_Custom : public PAPI
{
	public:
	void add_event (int event);
	void add_events (int *events_v, int events_c);
};

class PAPI_Preset : public PAPI
{
	protected:
	PAPI_Preset ();
	PAPI_Preset (int *events_v, int events_c);
};

class PAPI_Memory : public PAPI_Preset
{
	public:
	PAPI_Memory ();

	long long int loads();
	long long int stores();
};

class PAPI_CPI : public PAPI_Preset
{
	public:
	PAPI_CPI ();

	long long int instructions ();
	long long int cycles ();
	double cpi ();
	double ipc ();
};

class PAPI_Flops : public PAPI_Preset
{
	public:
	PAPI_Flops ();

	long long int flops();
	long long int cycles();
	double flops_per_cyc();
	double flops_per_sec();
};

class PAPI_Cache : public PAPI_Preset
{
	protected:
	PAPI_Cache ();

	public:
	virtual
	long long int accesses () = 0;

	virtual
	long long int misses () = 0;

	virtual
	double miss_rate () = 0;
};

class PAPI_L1 : public PAPI_Cache
{
	public:
	PAPI_L1 ();

	long long int accesses ();
	long long int misses ();
	double miss_rate ();
};

class PAPI_L2 : public PAPI_Cache
{
	public:
	PAPI_L2 ();

	long long int accesses ();
	long long int misses ();
	double miss_rate ();
};

class PAPI_InstPerByte : public PAPI_Preset
{
	public:
	PAPI_InstPerByte ();

	long long int instructions ();
	long long int ram_accesses ();
	long long int bytes_accessed ();
	double inst_per_byte ();
};

class PAPI_MulAdd : public PAPI_Preset
{
	public:
	PAPI_MulAdd ();

	long long int mults ();
	long long int divs ();
	long long int adds ();
	double balance ();
};

//	Stopwatch
class PAPI_Stopwatch
{
	bool _running;
	long long int _begin;
	long long int _end;
	long long int _partial;
	unsigned long long int _total;

	public:
	PAPI_Stopwatch();

	void start();
	void stop();
	void reset();

	//	getters
	bool running() const { return _running; }
	long long int partial() const { return _partial; }
	unsigned long long int total() const { return _total; }
};

#endif/*___PAPI_HPP___*/
