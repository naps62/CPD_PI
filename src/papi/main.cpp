#include <iostream>
#include <cstdlib>
#include <ctime>

#include "papi.hpp"

#define	ARRAY_SIZE	10000

using std::cout;
using std::endl;

int main ()
{
	int array[ ARRAY_SIZE ];
	int min, max;
	long long int sum, prd;
	unsigned i;
	PAPI p;

	srand( time(NULL) );

	for (i = 0; i < ARRAY_SIZE; ++i)
		array[i] = rand();

	p.add_event( PAPI_TOT_INS );

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
		<<	"PAPI_TOT_INS: "	<<	p[ PAPI_TOT_INS ]	<< endl;

	return 0;
}
