#include <iostream>
#include <cstdlib>
#include <climits>
#include <ctime>

#include <omp.h>

#include "papi.hpp"

#define	ARRAY_SIZE	10

using std::cout;
using std::endl;

int main ()
{
	int array[ ARRAY_SIZE ];
	int min, max;
	int *mins, *maxs;
	int tc;//	thread count
	int t;//	current thread
	long long int sum, prd;
	long long int *sums, *prds;
	unsigned i;
	PAPI p;

	min = INT_MAX;
	max = INT_MIN;
	sum = 0;
	prd = 1;

	srand( time(NULL) );

	for (i = 0; i < ARRAY_SIZE; ++i)
	{
		array[i] = rand() % ARRAY_SIZE + 1;
		cout
			<<	array[i]
			<<	endl;
	}

	p.add_event( PAPI_TOT_INS );
	p.add_event( PAPI_LD_INS );

	p.start();

	#pragma omp parallel default(shared) private(t)
	{
		#pragma omp master
		{
			tc = omp_get_num_threads();
			sums = new long long int[ tc ];
			prds = new long long int[ tc ];
			mins = new int[ tc ];
			maxs = new int[ tc ];
		}

		t = omp_get_thread_num();

		sums[t] = 0;
		prds[t] = 1;
		mins[t] = INT_MAX;
		maxs[t] = INT_MIN;

		#pragma omp for
		for (i = 0; i < ARRAY_SIZE; ++i)
		{
			sums[t] += array[i];
			prds[t] *= array[i];
			mins[t] = ( array[i] < mins[t] ) ? array[i] : mins[t];
			maxs[t] = ( array[i] > maxs[t] ) ? array[i] : maxs[t];
		}
	}

	sum = 0;
	prd = 1;
	min = INT_MAX;
	max = INT_MIN;

	for (t = 0; t < tc; ++t)
	{
		sum += sums[t];
		prd *= prds[t];
		min = ( array[i] < min ) ? array[i] : min;
		max = ( array[i] > max ) ? array[i] : max;
	}
	
	p.stop();

	cout
		<<	"sum: "	<<	sum	<<	endl
		<<	"prd: "	<<	prd	<<	endl
		<<	"max: "	<<	max	<<	endl
		<<	"min: "	<<	min	<<	endl
		<<	"PAPI_TOT_INS: "	<<	p[ PAPI_TOT_INS ]	<< endl
		<<	"PAPI_LD_INS: "	<<	p[ PAPI_LD_INS	]	<<	endl;

	return 0;
}
