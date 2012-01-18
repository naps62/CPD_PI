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
	//PAPI_Custom p;
	//PAPI_Memory p;
	//PAPI_CPI p;
	//PAPI_Flops p;
	//PAPI_L1 p;
	//PAPI_InstPerByte p;
	PAPI_MulAdd p;

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

	//p.add_event( PAPI_L2_DCA );
	//p.add_event( PAPI_L2_DCM );

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
//		<<	"PAPI_LD_INS: "	<<	p[ PAPI_LD_INS ]	<<	endl
//		<<	"PAPI_SR_INS: "	<<	p[ PAPI_SR_INS ]	<<	endl
		<<	"PAPI_FP_INS: "	<<	p[ PAPI_FP_INS ]	<<	endl
		<<	"PAPI_FML_INS: "	<<	p[ PAPI_FML_INS ]	<<	endl
		<<	"PAPI_FDV_INS: "	<<	p[ PAPI_FDV_INS	]	<<	endl
//		<<	"PAPI_TOT_CYC: "	<<	p[ PAPI_TOT_CYC	]	<<	endl
//		<<	"PAPI_FP_OPS: "	<<	p[ PAPI_FP_OPS ]	<<	endl
//		<<	"PAPI_L1_DCA: "	<<	p[ PAPI_L1_DCA ]	<<	endl
//		<<	"PAPI_L1_DCM: "	<<	p[ PAPI_L1_DCM ]	<<	endl
//		<<	"PAPI_L2_DCA: "	<<	p[ PAPI_L2_DCA ]	<<	endl
//		<<	"PAPI_L2_DCM: "	<<	p[ PAPI_L2_DCM ]	<<	endl
		<<	"Total time: "	<<	p.total_time()	<<	endl
//		<<	"CPI: "	<<	p.cpi()	<<	endl
//		<<	"IPC: "	<<	p.ipc()	<<	endl
//		<<	"Flops: "	<<	p.flops()	<<	endl
//		<<	"Flops/c: "	<<	p.flops_per_cyc()	<<	endl
//		<<	"Flops/s: "	<<	p.flops_per_sec()	<<	endl
//		<<	"L1 miss rate: "	<<	p.miss_rate()	<<	endl
//		<<	"Instructions: "	<<	p.instructions()	<<	endl
//		<<	"Bytes accessed: "	<<	p.bytes_accessed()	<<	endl
//		<<	"Intructions / RAM Byte: "	<<	p.inst_per_byte()	<<	endl
		<<	"FP multiplications: "	<<	p.mults()	<<	endl
		<<	"FP divisions: "	<<	p.divs()	<<	endl
		<<	"FP additions: "	<<	p.adds()	<<	endl
		<<	"MulAdd balance: "	<<	p.balance()	<<	endl
	;

	return 0;
}
