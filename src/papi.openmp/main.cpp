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
	long long int *tot_ins_v, *tot_cyc_v;
	long long int tot_ins, tot_cyc;
	unsigned i;

	min = INT_MAX;
	max = INT_MIN;
	sum = 0;
	prd = 1;
	tot_ins = 0;
	tot_cyc = 0;

	srand( time(NULL) );

	for (i = 0; i < ARRAY_SIZE; ++i)
	{
		array[i] = rand() % ARRAY_SIZE + 1;
		cout
			<<	array[i]
			<<	endl;
	}

	PAPI::init();

	#pragma omp parallel default(shared) private(t)
	{
		#pragma omp master
		{
			tc = omp_get_num_threads();
			sums = new long long int[ tc ];
			prds = new long long int[ tc ];
			mins = new int[ tc ];
			maxs = new int[ tc ];
			tot_ins_v = new long long int[ tc ];
			tot_cyc_v = new long long int[ tc ];
		}

		PAPI p;

		t = omp_get_thread_num();

		sums[t] = 0;
		prds[t] = 1;
		mins[t] = INT_MAX;
		maxs[t] = INT_MIN;

		p.add_event( PAPI_TOT_INS );
		p.add_event( PAPI_TOT_CYC );

		p.start();

		#pragma omp for
		for (i = 0; i < ARRAY_SIZE; ++i)
		{
			sums[t] += array[i];
			prds[t] *= array[i];
			mins[t] = ( array[i] < mins[t] ) ? array[i] : mins[t];
			maxs[t] = ( array[i] > maxs[t] ) ? array[i] : maxs[t];
		}

		p.stop();

		tot_ins_v[t] = p[ PAPI_TOT_INS ];
		tot_cyc_v[t] = p[ PAPI_TOT_CYC ];
	}

	for (t = 0; t < tc; ++t)
	{
		sum += sums[t];
		prd *= prds[t];
		min = ( mins[t] < min ) ? mins[t] : min;
		max = ( maxs[t] > max ) ? maxs[t] : max;
		tot_ins += tot_ins_v[t];
		tot_cyc += tot_cyc_v[t];
	}

	cout
		<<	"sum: "	<<	sum	<<	endl
		<<	"prd: "	<<	prd	<<	endl
		<<	"max: "	<<	max	<<	endl
		<<	"min: "	<<	min	<<	endl
		<<	"PAPI_TOT_INS: "	<<	tot_ins	<< endl
		<<	"PAPI_TOT_CYC: "	<<	tot_cyc	<<	endl;
	
	PAPI::shutdown();

	return 0;
}
