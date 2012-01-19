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

	//	total values
	long long int sum;
	long long int prd;
	long long int tot_cyc;
	long long int tot_ins;
	long long int ld_ins;
	long long int sr_ins;
	long long int fml_ins;
	long long int fdv_ins;
	long long int fp_ops;
	long long int l1_dca;
	long long int l1_dcm;
	long long int l2_dca;
	long long int l2_dcm;
	long long int tot_tm;
	unsigned i;

	//	thread values arrays
	long long int *sums;
	long long int *prds;
	long long int *tot_cyc_v;
	long long int *tot_ins_v;
	long long int *ld_ins_v;
	long long int *sr_ins_v;
	long long int *fml_ins_v;
	long long int *fdv_ins_v;
	long long int *fp_ops_v;
	long long int *l1_dca_v;
	long long int *l1_dcm_v;
	long long int *l2_dca_v;
	long long int *l2_dcm_v;
	long long int *tot_tm_v;

	min = INT_MAX;
	max = INT_MIN;
	sum = 0;
	prd = 1;

	tot_cyc = 0;
	tot_ins = 0;
	ld_ins = 0;
	sr_ins = 0;
	fml_ins = 0;
	fdv_ins = 0;
	fp_ops = 0;
	l1_dca = 0;
	l1_dcm = 0;
	l2_dca = 0;
	l2_dcm = 0;

	tot_tm = 0;

	//	read arguments
	if (argc > 1)
		mode = atoi( argv[1] );
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
				case 0:
					//	Loads / Stores
					ld_ins_v = new long long int [ tc ];
					sr_ins_v = new long long int [ tc ];
					break;

				case 2:
					//	Flops
					tot_cyc_v = new long long int[ tc ];
					fp_ops_v = new long long int [ tc ];
					break;
				
				case 3:
					//	L1
					l1_dca_v = new long long int[ tc ];
					l1_dcm_v = new long long int[ tc ];
					break;

				case 4:
					//	L2 accesses
					l2_dca_v = new long long int[ tc ];
					break;

				case 5:
					//	IPB
					tot_ins_v = new long long int[ tc ];
					l2_dcm_v = new long long int[ tc ];
					break;

				case 6:
					//	MAB_ML
					fml_ins_v = new long long int[ tc ];
					break;

				case 7:
					//	MAB_DV
					fdv_ins_v = new long long int[ tc ];
					break;

				case 1:
				default:
					//	CPI
					tot_ins_v = new long long int[ tc ];
					tot_cyc_v = new long long int[ tc ];
			}
		}

		PAPI *p;
		PAPI_Custom *pc;
		PAPI_Memory *pm;
		PAPI_CPI *pcpi;
		PAPI_Flops *pf;
		PAPI_L1 *pl1;
		PAPI_InstPerByte *pipb;

		p = NULL;
		pc = NULL;
		pm = NULL;
		pcpi = NULL;
		pf = NULL;
		pl1 = NULL;
		pipb = NULL;

		switch ( mode )
		{
			case 0:
				//	Loads / Stores
				pm = new PAPI_Memory();
				p = pm;
				break;

			case 2:
				//	Flops
				pf = new PAPI_Flops();
				p = pf;
				break;

			case 3:
				//	L1
				pl1 = new PAPI_L1();
				p = pl1;
				break;

			case 4:
				//	L2 accesses
				pc = new PAPI_Custom();
				pc->add_event( PAPI_L2_DCA );
				p = pc;
				break;

			case 5:
				//	IPB
				pipb = new PAPI_InstPerByte();
				p = pipb;
				break;

			case 6:
				//	MAB_ML
				pc = new PAPI_Custom();
				pc->add_event( PAPI_FML_INS );
				p = pc;
				break;

			case 7:
				//	MAB_DV
				pc = new PAPI_Custom();
				pc->add_event( PAPI_FDV_INS );
				p = pc;
				break;

			case 1:
			default:
				//	CPI
				pcpi = new PAPI_CPI();
				p = pcpi;
		}

		t = omp_get_thread_num();

		sums[t] = 0;
		prds[t] = 1;
		mins[t] = INT_MAX;
		maxs[t] = INT_MIN;

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
			case 0:
				//	Loads / Stores
				ld_ins_v[t] = pm->loads();
				sr_ins_v[t] = pm->stores();
				break;

			case 2:
				//	Flops
				tot_cyc_v[t] = pf->cycles();
				fp_ops_v[t] = pf->flops();
				break;

			case 3:
				//	L1
				l1_dca_v[t] = pl1->accesses();
				l1_dcm_v[t] = pl1->misses();
				break;

			case 4:
				//	L2 accesses
				l2_dca_v[t] = pc->get( PAPI_L2_DCA );
				break;

			case 5:
				//	IPB
				tot_ins_v[t] = pipb->instructions();
				l2_dcm_v[t] = pipb->ram_accesses();
				break;

			case 6:
				//	MAB_ML
				fml_ins_v[t] = pc->get( PAPI_FML_INS );
				break;

			case 7:
				//	MAB_DV
				fdv_ins_v[t] = pc->get( PAPI_FDV_INS );
				break;

			case 1:
			default:
				//	CPI
				tot_ins_v[t] = pcpi->instructions();
				tot_cyc_v[t] = pcpi->cycles();
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
				case 0:
					//	Loads / Stores
					cout
						<<	"Loads: "
						<<	ld_ins_v[t]
						<<	endl
						<<	"Stores: "
						<<	sr_ins_v[t]
						<<	endl;
					break;

				case 2:
					//	Flops
					cout
						<<	"Cycles: "
						<<	tot_cyc_v[t]
						<<	endl
						<<	"Flops: "
						<<	fp_ops_v[t]
						<<	endl;
					break;

				case 3:
					//	L1
					cout
						<<	"L1 accesses: "
						<<	l1_dca_v[t]
						<<	endl
						<<	"L1 misses: "
						<<	l1_dcm_v[t]
						<<	endl;
					break;

				case 4:
					//	L2 accesses
					cout
						<<	"L2 accesses: "
						<<	l2_dca_v[t]
						<<	endl;
					break;

				case 5:
					//	IPB
					cout
						<<	"Total instructions: "
						<<	tot_ins_v[t]
						<<	endl
						<<	"L2 misses: "
						<<	l2_dcm_v[t]
						<<	endl;
					break;

				case 6:
					//	MAB_ML
					cout
						<<	"Multiplications: "
						<<	fml_ins_v[t]
						<<	endl;
					break;

				case 7:
					//	MAB_DV
					cout
						<<	"Divisions: "
						<<	fdv_ins_v[t]
						<<	endl;
					break;

				case 1:
				default:
					//	CPI
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
			case 0:
				//	Loads / Stores
				ld_ins += ld_ins_v[t];
				sr_ins += sr_ins_v[t];
				break;

			case 2:
				//	Flops
				tot_cyc += tot_cyc_v[t];
				fp_ops += fp_ops_v[t];
				break;

			case 3:
				//	L1
				l1_dca += l1_dca_v[t];
				l1_dcm += l1_dcm_v[t];
				break;

			case 4:
				//	L2 accesses
				l2_dca += l2_dca_v[t];
				break;

			case 5:
				//	IPB
				tot_ins += tot_ins_v[t];
				l2_dcm += l2_dcm_v[t];
				break;

			case 6:
				//	MAB_ML
				fml_ins += fml_ins_v[t];
				break;

			case 7:
				//	MAB_DV
				fdv_ins += fdv_ins_v[t];
				break;

			case 1:
			default:
				//	CPI
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
		case 0:
			//	Loads / Stores
			cout
				<<	"Load instructions: "
				<<	ld_ins
				<<	endl
				<<	"Store instructions: "
				<<	sr_ins
				<<	endl;
			break;

		case 2:
			//	Flops
			cout
				<<	"Total cycles: "
				<<	tot_cyc
				<<	endl
				<<	"FP operations: "
				<<	fp_ops
				<<	endl;
			break;

		case 3:
			//	L1
			cout
				<<	"L1 accesses: "
				<<	l1_dca
				<<	endl
				<<	"L1 misses: "
				<<	l1_dcm
				<<	endl;
			break;

		case 4:
			//	L2 accesses
			cout
				<<	"L2 accesses: "
				<<	l2_dca
				<<	endl;
			break;

		case 5:
			//	IPB
			cout
				<<	"Total instructions: "
				<<	tot_ins
				<<	endl
				<<	"L2 misses: "
				<<	l2_dcm
				<<	endl;
			break;

		case 6:
			//	MAB_ML
			cout
				<<	"Multiplications: "
				<<	fml_ins
				<<	endl;
			break;

		case 7:
			//	MAB_DV
			cout
				<<	"Divisions: "
				<<	fdv_ins
				<<	endl;
			break;

		case 1:
		default:
			//	CPI
			cout
				<<	"Total instructions:: "
				<<	tot_ins
				<<	endl
				<<	"Total cycles: "
				<<	tot_cyc
				<<	endl;
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
		case 0:
			//	Loads / Stores
			delete ld_ins_v;
			delete sr_ins_v;
			break;

		case 2:
			//	Flops
			delete tot_cyc_v;
			delete fp_ops_v;
			break;

		case 3:
			//	L1
			delete l1_dca_v;
			delete l1_dcm_v;
			break;

		case 4:
			//	L2 accesses
			delete l2_dca_v;
			break;

		case 5:
			//	IPB
			delete tot_ins_v;
			delete l2_dcm_v;
			break;

		case 6:
			//	MAB_ML
			delete fml_ins_v;
			break;

		case 7:
			//	MAB_DV
			delete fdv_ins_v;
			break;

		case 1:
		default:
			//	CPI
			delete tot_ins_v;
			delete tot_cyc_v;
	}

	return 0;
}
