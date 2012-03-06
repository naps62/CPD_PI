#ifndef ___PAPI_COUNTER___
#define ___PAPI_COUNTER___

#include <map>

namespace papi
{
	class Counter
	{
		int             _set;
		long long int * _values;
		map<int,long long int> counters;

#endif//___PAPI_COUNTER___
