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
	PAPI_CPI p;

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

	//p.add_event( PAPI_TOT_INS );
	//p.add_event( PAPI_LD_INS );

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
		<<	"PAPI_TOT_INS: "	<<	p[ PAPI_TOT_INS ]	<<	endl
		<<	"PAPI_TOT_CYC: "	<<	p[ PAPI_TOT_CYC	]	<<	endl
		<<	"CPI: "	<<	p.cpi()	<<	endl
		<<	"IPC: "	<<	p.ipc()	<<	endl
	;

	return 0;
}
