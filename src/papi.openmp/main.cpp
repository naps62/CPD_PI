#include <iostream>
#include <cstdlib>
#include <climits>
#include <ctime>

#include <omp.h>

#include "papi.hpp"

#define	ARRAY_SIZE	10

using std::cout;
using std::endl;

int main (int argc, char **argv)
{
	int array[ ARRAY_SIZE ];
	int min, max;
	int *mins, *maxs;
	int mode;								//	PAPI mode
	int tc;//	thread count
	int t;//	current thread
	long long int sum, prd;
	long long int *sums, *prds;
	long long int *tot_ins_v, *tot_cyc_v;
	long long int tot_ins, tot_cyc;
	long long int *tot_tm_v;
	long long int tot_tm;
	unsigned i;

	min = INT_MAX;
	max = INT_MIN;
	sum = 0;
	prd = 1;
	tot_ins = 0;
	tot_cyc = 0;
	tot_tm = 0;

	//	read arguments
	if (argc > 1)
		mode = atoi( argv[0] );
	else
		mode = -1;
	
	srand( time(NULL) );

	for (i = 0; i < ARRAY_SIZE; ++i)
	{
		array[i] = rand() % ARRAY_SIZE + 1;
		cout
			<<	array[i]
			<<	endl;
	}

	PAPI::init_threads();

	#pragma omp parallel default(shared) private(t)
	{
		#pragma omp master
		{
			tc = omp_get_num_threads();
			sums = new long long int[ tc ];
			prds = new long long int[ tc ];
			mins = new int[ tc ];
			maxs = new int[ tc ];
			tot_tm_v = new long long int[ tc ];

			switch ( mode )
			{
				default:
				tot_ins_v = new long long int[ tc ];
				tot_cyc_v = new long long int[ tc ];
			}
		}

		PAPI *p;

		switch ( mode )
		{
			default:
				PAPI_Custom *pc;
				pc = new PAPI_Custom();
				pc->add_event( PAPI_TOT_INS );
				pc->add_event( PAPI_TOT_CYC );
				p = pc;
		}

		t = omp_get_thread_num();

		sums[t] = 0;
		prds[t] = 1;
		mins[t] = INT_MAX;
		maxs[t] = INT_MIN;

//		p.add_event( PAPI_TOT_INS );
//		p.add_event( PAPI_TOT_CYC );

		p->start();

		#pragma omp for
		for (i = 0; i < ARRAY_SIZE; ++i)
		{
			sums[t] += array[i];
			prds[t] *= array[i];
			mins[t] = ( array[i] < mins[t] ) ? array[i] : mins[t];
			maxs[t] = ( array[i] > maxs[t] ) ? array[i] : maxs[t];
		}

		p->stop();

		tot_tm_v[t] = p->last_time();
		switch ( mode )
		{
			default:
			tot_ins_v[t] = p->get( PAPI_TOT_INS );
			tot_cyc_v[t] = p->get( PAPI_TOT_CYC );
		}

	delete p;

		#pragma omp barrier

		#pragma omp critical
		{
			cout
				<<	'<'
				<<	t
				<<	'>'
				<<	endl;
			switch ( mode )
			{
				default:
				cout
					<<	"\tPAPI_TOT_INS: "
					<<	tot_ins_v[t]
					<<	endl
					<<	"\tPAPI_TOT_CYC: "
					<<	tot_cyc_v[t]
					<<	endl;
			}
			cout
				<<	"\telapsed time: "
				<<	tot_tm_v[t]
				<<	endl;
		}
	}

	for (t = 0; t < tc; ++t)
	{
		sum += sums[t];
		prd *= prds[t];
		min = ( mins[t] < min ) ? mins[t] : min;
		max = ( maxs[t] > max ) ? maxs[t] : max;
		tot_tm += tot_tm_v[t];
		switch ( mode )
		{
			default:
			tot_ins += tot_ins_v[t];
			tot_cyc += tot_cyc_v[t];
		}
	}

	cout
		<<	"sum: "	<<	sum	<<	endl
		<<	"prd: "	<<	prd	<<	endl
		<<	"max: "	<<	max	<<	endl
		<<	"min: "	<<	min	<<	endl;
	switch ( mode )
	{
		default:
		cout
			<<	"PAPI_TOT_INS: "	<<	tot_ins	<< endl
			<<	"PAPI_TOT_CYC: "	<<	tot_cyc	<<	endl;
	}
	cout
		<<	"total time: "	<<	tot_tm	<<	endl;
	
	//	cleanup
	PAPI::shutdown();

	delete sums;
	delete prds;
	delete mins;
	delete maxs;
	delete tot_tm_v;
	switch ( mode )
	{
		default:
		delete tot_ins_v;
		delete tot_cyc_v;
	}

	return 0;
}
