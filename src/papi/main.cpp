#include <iostream>
#include <cstdlib>
#include <climits>
#include <ctime>

#include "papi.hpp"

#define	ARRAY_SIZE	10

using std::cout;
using std::endl;

int main ()
{
	int array[ ARRAY_SIZE ];
	int min, max;
	long long int sum, prd;
	unsigned i;
	//PAPI_CPI p;
	PAPI_Flops p;

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

	p.start();

	for (i = 0; i < ARRAY_SIZE; ++i)
	{
		sum += array[i];
		prd *= array[i];
		max = ( array[i] > max ) ? array[i] : max;
		min = ( array[i] < min ) ? array[i] : min;
	}
	
	p.stop();

	cout
		<<	"sum: "	<<	sum	<<	endl
		<<	"prd: "	<<	prd	<<	endl
		<<	"max: "	<<	max	<<	endl
		<<	"min: "	<<	min	<<	endl
//		<<	"PAPI_TOT_INS: "	<<	p[ PAPI_TOT_INS ]	<<	endl
		<<	"PAPI_TOT_CYC: "	<<	p[ PAPI_TOT_CYC	]	<<	endl
		<<	"PAPI_FP_OPS: "	<<	p[ PAPI_FP_OPS ]	<<	endl
		<<	"Total time: "	<<	p.total_time()	<<	endl
//		<<	"CPI: "	<<	p.cpi()	<<	endl
//		<<	"IPC: "	<<	p.ipc()	<<	endl
		<<	"Flops: "	<<	p.flops()	<<	endl
		<<	"Flops/c: "	<<	p.flops_per_cyc()	<<	endl
		<<	"Flops/s: "	<<	p.flops_per_sec()	<<	endl
	;

	return 0;
}
