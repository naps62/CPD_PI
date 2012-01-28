#include <iostream>
#include <limits>

#include "FVLib.h"


//	BEGIN CONSTANTS

/**
 * OpenMP [f]actor
 * Used with the number of processors to calculate the number of threads:
 * pc -> processor count
 * tc -> thread count
 * => tc = pc * f
 * This allows the existence of more threads than the hardware is capable, therefore making possible for some threads to step in while others wait on resources (based on GPU approach).
 */
#define	OMP_FCT_CF	1

 /**
  * PAPI
  */
#if defined	(PAPI_MEASURE_CPI)	\
 || defined	(PAPI_MEASURE_MEMORY)	\
 || defined	(PAPI_MEASURE_FLOPS)	\
 || defined	(PAPI_MEASURE_L1)	\
 ||	defined	(PAPI_MEASURE_L2DCA)	\
 ||	defined	(PAPI_MEASURE_IPB)	\
 ||	defined	(PAPI_MEASURE_MBAML)	\
 ||	defined	(PAPI_MEASURE_MBADV)
#define	PAPI_MEASURE
#define PAPI_MEASURE_TIME
#endif

#ifdef	PAPI_MEASURE
#include "papi.hpp"
#endif

//	END CONSTANTS

//	BEGIN TYPES

/*
	Parameters: holds the data from the parameter file
*/
typedef
struct _parameters
{
	struct
	{
		string mesh;
		string velocity;
		struct
		{
			string initial;
			string output;
		} polution;
	} filenames;
	struct
	{
		double final;
	} time;
	struct
	{
		int jump;
	} iterations;
	struct
	{
		double threshold;
	} computation;
}
Parameters;

//	END TYPES

//	BEGIN GLOBAL VARIABLES

int tc;						//	thread count
#if defined		(PAPI_MEASURE_CPI)
long long int tot_cyc;		//	process total cycles
long long int tot_ins;		//	process total instructions
#elif defined		(PAPI_MEASURE_MEMORY)
long long int ld_ins;		//	process total load instructions
long long int sr_ins;		//	process total store instructions
#elif defined	(PAPI_MEASURE_FLOPS)
long long int tot_cyc;		//	process total cycles
long long int fp_ops;		//	process total FP operations
#elif defined	(PAPI_MEASURE_L1)
long long int l1_dca;		//	process total L1 data cache accesses
long long int l1_dcm;		//	process total L1 data cache misses
#elif defined	(PAPI_MEASURE_L2DCA)
long long int l2_dca;		//	process total L2 data cache accesses
#elif defined	(PAPI_MEASURE_IPB)
long long int tot_ins;		//	process total instructions
long long int l2_dcm;		//	process total L2 data cache misses
#elif defined	(PAPI_MEASURE_MBAML)
long long int fp_ins;		//	process total FP instructions
long long int fml_ins;		//	process total FP multiplication instructions
#elif defined	(PAPI_MEASURE_MBADV)
long long int fp_ins;		//	process total FP instructions
long long int fdv_ins;		//	process total FP division instructions
#endif
#ifdef	PAPI_MEASURE
long long int tot_ns;		//	process total time in nano-seconds
#endif//	PAPI_MEASURE

#ifdef	PAPI_MEASURE_TIME
							//	compute_flux
unsigned
long long int cf_tot_ns;	//	compute_flux total useful time (nano seconds)
unsigned
long long int cf_max_ns;	//	compute_flux maximum useful time
unsigned
long long int cf_min_ns;	//	compute_flux minimum useful time
unsigned
long int cf_count;			//	compute_flux call count
#endif//	PAPI_MEASURE_TIME

							//	vectors
double *max_vel_v;			//	threads max velocity vector
#if defined		(PAPI_MEASURE_CPI)
long long int *tot_cyc_v;	//	threads total cycles vector
long long int *tot_ins_v;	//	threads total instructions vector
#elif defined		(PAPI_MEASURE_MEMORY)
long long int *ld_ins_v;	//	threads load instructions vector
long long int *sr_ins_v;	//	threads store instructions vector
#elif defined	(PAPI_MEASURE_FLOPS)
long long int *tot_cyc_v;	//	threads total cycles vector
long long int *fp_ops_v;	//	threads FP operations vector
#elif defined	(PAPI_MEASURE_L1)
long long int *l1_dca_v;	//	threads L1 data cache accesses vector
long long int *l1_dcm_v;	//	threads L1 data cache misses vector
#elif defined	(PAPI_MEASURE_L2DCA)
long long int *l2_dca_v;	//	threads L2 data cache accesses vector
#elif defined	(PAPI_MEASURE_IPB)
long long int *tot_ins_v;	//	threads total instructions vector
long long int *l2_dcm_v;	//	threads L2 data cache misses vector
#elif defined	(PAPI_MEASURE_MBAML)
long long int *fp_ins_v;	//	threads FP total instructions
long long int *fml_ins_v;	//	threads FP multiplication instructions
#elif defined	(PAPI_MEASURE_MBADV)
long long int *fp_ins_v;	//	threads FP total instructions
long long int *fdv_ins_v;	//	threads FP division instructions
#endif
#ifdef	PAPI_MEASURE
long long int *tot_ns_v;	//	threads total time in nano-seconds
#endif

//	END GLOBAL VARIABLES

//	BEGIN FUNCTIONS

/*
	Computes the resulting flux in every edge
*/
double compute_flux(
	FVMesh2D& mesh,
	FVVect<double>& polution,
	FVVect<FVPoint2D<double> >& velocity,
	FVVect<double>& flux,
	double dc)								//	Dirichlet condition
{
	double dt;								//	elapsed time
	double p_left;							//	polution in the left face
	double p_right;							//	polution in the right face
	int i_left;								//	index of the left face
	int i_right;							//	index of the right face
	int t;									//	current thread number
	unsigned e;								//	edge iteration variable
	unsigned es;							//	total number of edges
	FVPoint2D<double> v_left;				//	velocity in the left face
	FVPoint2D<double> v_right;				//	velocity in the right face
	double v;								//	resulting velocity
	FVEdge2D *edge;							//	current edge

	double max_vel;							//	maximum calculated velocity
#ifdef	PAPI_MEASURE
#if defined		(PAPI_MEASURE_CPI)
	long long int tot_cyc_s;				//	total cycles (sequential zone)
	long long int tot_ins_s;				//	total instructions (sequential zone)
#elif defined		(PAPI_MEASURE_MEMORY)
	long long int ld_ins_s;					//	load instructions (sequential zone)
	long long int sr_ins_s;					//	store instructions (sequential zone)
#elif defined	(PAPI_MEASURE_FLOPS)
	long long int tot_cyc_s;				//	total cycles (sequential zone)
	long long int fp_ops_s;					//	FP operations (sequential zone)
#elif defined	(PAPI_MEASURE_L1)
	long long int l1_dca_s;					//	L1 data cache accesses (sequential)
	long long int l1_dcm_s;					//	L1 data cache misses (sequential)
#elif defined	(PAPI_MEASURE_L2DCA)
	long long int l2_dca_s;					//	L2 data cache accesses (sequential)
#elif defined	(PAPI_MEASURE_IPB)
	long long int tot_ins_s;				//	total instructions (sequential)
	long long int l2_dcm_s;					//	L2 data cache misses (sequential)
#elif defined	(PAPI_MEASURE_MBAML)
	long long int fp_ins_s;					//	FP total instructions (sequential)
	long long int fml_ins_s;				//	FP multiplication instructions (sequential)
#elif defined	(PAPI_MEASURE_MBADV)
	long long int fp_ins_s;					//	FP total instructions (sequential)
	long long int fdv_ins_s;				//	FP division instructions (sequential)
#endif
	long long int tot_ns_s;					//	total time in nano-seconds (sequential zone)
#endif//	PAPI_MEASURE
#if defined		(PAPI_MEASURE_TIME)
	PAPI_Stopwatch sw;						//	a proper time measurer
#endif
											//	PAPI specific eventset
#if defined		(PAPI_MEASURE_CPI)
	PAPI_CPI *p;							
#elif defined	(PAPI_MEASURE_MEMORY)
	PAPI_Memory *p;
#elif defined	(PAPI_MEASURE_FLOPS)
	PAPI_Flops *p;							
#elif defined	(PAPI_MEASURE_L1)
	PAPI_L1 *p;
#elif defined	(PAPI_MEASURE_L2DCA)
	PAPI_Custom *p;
#elif defined	(PAPI_MEASURE_IPB)
	PAPI_InstPerByte *p;
#elif defined	(PAPI_MEASURE_MBAML)
	PAPI_Custom *p;
#elif defined	(PAPI_MEASURE_MBADV)
	PAPI_Custom *p;
#endif

#ifdef	PAPI_MEASURE_TIME
	sw.start();
#endif//	PAPI_MEASURE_TIME
	es = mesh.getNbEdge();
#ifdef	PAPI_MEASURE_TIME
	sw.stop();
#endif//	PAPI_MEASURE_TIME

#ifdef	PAPI_MEASURE
	#pragma omp parallel	\
		default(shared)	\
		num_threads(tc)	\
		private(t,e,edge,i_left,v_left,p_left,i_right,v_right,p_right,v,p)
#else//		PAPI_MEASURE
	#pragma omp parallel	\
		default(shared)	\
		num_threads(tc)	\
		private(t,e,edge,i_left,v_left,p_left,i_right,v_right,p_right,v)
#endif//	PAPI_MEASURE
	{
#if defined		(PAPI_MEASURE_CPI)
		p = new PAPI_CPI();
#elif defined		(PAPI_MEASURE_MEMORY)
		p = new PAPI_Memory();
#elif defined	(PAPI_MEASURE_FLOPS)
		p = new PAPI_Flops();
#elif defined	(PAPI_MEASURE_L1)
		p = new PAPI_L1();
#elif defined	(PAPI_MEASURE_L2DCA)
		p = new PAPI_Custom();
		p->add_event( PAPI_L2_DCA );
#elif defined	(PAPI_MEASURE_IPB)
		p = new PAPI_InstPerByte();
#elif defined	(PAPI_MEASURE_MBAML)
		p = new PAPI_Custom();
		p->add_event( PAPI_FP_INS );
		p->add_event( PAPI_FML_INS );
#elif defined	(PAPI_MEASURE_MBADV)
		p = new PAPI_Custom();
		p->add_event( PAPI_FP_INS );
		p->add_event( PAPI_FDV_INS );
#endif

		t = omp_get_thread_num();

		max_vel_v[t] = numeric_limits<double>::min();

#ifdef	PAPI_MEASURE_TIME
#pragma omp barrier
#pragma omp master
		sw.start();
#endif//	PAPI_MEASURE_TIME

		//	start measure
#ifdef	PAPI_MEASURE
		p->start();
#endif

		#pragma omp for
		for (e = 0; e < es; ++e)
		{
			edge = mesh.getEdge(e);
			i_left = edge->leftCell->label - 1;
			v_left = velocity[ i_left ];
			p_left = polution[ i_left ];
			if ( edge->rightCell ) 
			{
				i_right = edge->rightCell->label - 1;
				v_right = velocity[ i_right ];
				p_right = polution[ i_right ];
			}
			else
			{
				v_right = v_left;
				p_right = dc;
			} 
			v = ( v_left + v_right ) * 0.5 * edge->normal; 
			max_vel_v[t] = ( v > max_vel_v[t] ) ? v : max_vel_v[t];
			if ( v < 0 )
				flux[ edge->label - 1 ] = v * p_right;
			else
				flux[ edge->label - 1 ] = v * p_left;
		}

		//	stop measure
#ifdef	PAPI_MEASURE
		p->stop();
#endif

#ifdef	PAPI_MEASURE_TIME
#pragma omp barrier
#pragma omp master
		sw.stop();
#endif//	PAPI_MEASURE_TIME

		//	get values
#if defined		(PAPI_MEASURE_CPI)
		tot_cyc_v[t] = p->cycles();
		tot_ins_v[t] = p->instructions();
#elif defined	(PAPI_MEASURE_MEMORY)
		ld_ins_v[t] = p->loads();
		sr_ins_v[t] = p->stores();
#elif defined	(PAPI_MEASURE_FLOPS)
		tot_cyc_v[t] = p->cycles();
		fp_ops_v[t] = p->flops();
#elif defined	(PAPI_MEASURE_L1)
		l1_dca_v[t] = p->accesses();
		l1_dca_v[t] = p->misses();
#elif defined	(PAPI_MEASURE_L2DCA)
		l2_dca_v[t] = p->get( PAPI_L2_DCA );
#elif defined	(PAPI_MEASURE_IPB)
		tot_ins_v[t] = p->instructions();
		l2_dcm_v[t] = p->ram_accesses();
#elif defined	(PAPI_MEASURE_MBAML)
		fp_ins_v[t] = p->get( PAPI_FP_INS );
		fml_ins_v[t] = p->get( PAPI_FML_INS );
#elif defined	(PAPI_MEASURE_MBADV)
		fp_ins_v[t] = p->get( PAPI_FP_INS );
		fdv_ins_v[t] = p->get( PAPI_FDV_INS );
#endif
#ifdef	PAPI_MEASURE
		tot_ns_v[t] = p->last_time();

		//	cleanup
		delete p;
#endif
	}

	//	set sequential eventset
#if defined		(PAPI_MEASURE_CPI)
	p = new PAPI_CPI();
#elif defined	(PAPI_MEASURE_MEMORY)
	p = new PAPI_Memory();
#elif defined	(PAPI_MEASURE_FLOPS)
	p = new PAPI_Flops();
#elif defined	(PAPI_MEASURE_L1)
	p = new PAPI_L1();
#elif defined	(PAPI_MEASURE_L2DCA)
	p = new PAPI_Custom();
	p->add_event( PAPI_L2_DCA );
#elif defined	(PAPI_MEASURE_IPB)
	p = new PAPI_InstPerByte();
#elif defined	(PAPI_MEASURE_MBAML)
	p = new PAPI_Custom();
	p->add_event( PAPI_FP_INS );
	p->add_event( PAPI_FML_INS );
#elif defined	(PAPI_MEASURE_MBADV)
	p = new PAPI_Custom();
	p->add_event( PAPI_FP_INS );
	p->add_event( PAPI_FDV_INS );
#endif


#ifdef	PAPI_MEASURE_TIME
	sw.start();
#endif

	//	start sequential measure
#ifdef	PAPI_MEASURE
	p->start();
#endif

	max_vel = numeric_limits<double>::min();
	for (t = 0; t < tc; ++t)
		max_vel = ( max_vel_v[t] > max_vel ) ? max_vel_v[t] : max_vel;

	dt = 1.0 / abs( max_vel );

	//	stop sequential measure
#ifdef	PAPI_MEASURE
	p->stop();
#endif

#ifdef	PAPI_MEASURE_TIME
	sw.stop();
#endif//	PAPI_MEASURE_TIME

	//	get sequential values
#if defined		(PAPI_MEASURE_CPI)
	tot_cyc_s = p->cycles();
	tot_ins_s = p->instructions();
#elif defined	(PAPI_MEASURE_MEMORY)
	ld_ins_s = p->loads();
	sr_ins_s = p->stores();
#elif defined	(PAPI_MEASURE_FLOPS)
	tot_cyc_s = p->cycles();
	fp_ops_s = p->flops();
#elif defined	(PAPI_MEASURE_L1)
	l1_dca_s = p->accesses();
	l1_dcm_s = p->misses();
#elif defined	(PAPI_MEASURE_L2DCA)
	l2_dca_s = p->get( PAPI_L2_DCA );
#elif defined	(PAPI_MEASURE_IPB)
	tot_ins_s = p->instructions();
	l2_dcm_s = p->ram_accesses();
#elif defined	(PAPI_MEASURE_MBAML)
	fp_ins_s = p->get( PAPI_FP_INS );
	fml_ins_s = p->get( PAPI_FML_INS );
#elif defined	(PAPI_MEASURE_MBADV)
	fp_ins_s = p->get( PAPI_FP_INS );
	fdv_ins_s = p->get( PAPI_FDV_INS );
#endif
#ifdef	PAPI_MEASURE
	tot_ns_s = p->last_time();

	//	cleanup
	delete p;
#endif

	//	gather PAPI results
#ifdef	PAPI_MEASURE
	for (t = 0; t < tc; ++t)
	{
#if defined			(PAPI_MEASURE_CPI)
		tot_cyc += tot_cyc_v[t];
		tot_ins += tot_ins_v[t];
#elif defined		(PAPI_MEASURE_MEMORY)
		ld_ins += ld_ins_v[t];
		sr_ins += sr_ins_v[t];
#elif defined	(PAPI_MEASURE_FLOPS)
		tot_cyc += tot_cyc_v[t];
		fp_ops += fp_ops_v[t];
#elif defined	(PAPI_MEASURE_L1)
		l1_dca += l1_dca_v[t];
		l1_dcm += l1_dcm_v[t];
#elif defined	(PAPI_MEASURE_L2DCA)
		l2_dca += l2_dca_v[t];
#elif defined	(PAPI_MEASURE_IPB)
		tot_ins += tot_ins_v[t];
		l2_dcm += l2_dcm_v[t];
#elif defined	(PAPI_MEASURE_MBAML)
		fp_ins += fp_ins_v[t];
		fml_ins += fml_ins_v[t];
#elif defined	(PAPI_MEASURE_MBADV)
		fp_ins += fp_ins_v[t];
		fdv_ins += fdv_ins_v[t];
#endif
		tot_ns += tot_ns_v[t];
	}
#if defined		(PAPI_MEASURE_CPI)
	tot_cyc += tot_cyc_s;
	tot_ins += tot_ins_s;
#elif defined		(PAPI_MEASURE_MEMORY)
	ld_ins += ld_ins_s;
	sr_ins += sr_ins_s;
#elif defined	(PAPI_MEASURE_FLOPS)
	tot_cyc += tot_cyc_s;
	fp_ops += fp_ops_s;			
#elif defined	(PAPI_MEASURE_L1)
	l1_dca += l1_dca_s;
	l1_dcm += l1_dcm_s;
#elif defined	(PAPI_MEASURE_L2DCA)
	l2_dca += l2_dca_s;
#elif defined	(PAPI_MEASURE_IPB)
	tot_ins += tot_ins_s;
	l2_dcm += l2_dcm_s;
#elif defined	(PAPI_MEASURE_MBAML)
	fp_ins += fp_ins_s;
	fml_ins += fml_ins_s;
#elif defined	(PAPI_MEASURE_MBADV)
	fp_ins += fp_ins_s;
	fdv_ins += fdv_ins_s;
#endif
	tot_ns += tot_ns_s;
#endif//	PAPI_MEASURE

#ifdef	PAPI_MEASURE_TIME
	cf_ns = sw.total();
	cf_tot_ns += cf_ns;
	cf_max_ns = ( cf_ns > cf_max_ns ) ? cf_ns : cf_max_ns;
	cf_min_ns = ( cf_ns < cf_min_ns ) ? cf_ns : cf_min_ns;
	cf_count += 1;
#endif//	PAPI_MEASURE_TIME
	
	return dt;
}

/*
	Updates the polution values based on the flux through every edge.
*/
void update(
	FVMesh2D& mesh,
	FVVect<double>& polution,
	FVVect<double>& flux,
	double dt)
{
	FVEdge2D *edge;

	int es = mesh.getNbEdge();
	for (int e = 0; e < es; ++e)
	{
		edge = mesh.getEdge(e);
		polution[ edge->leftCell->label - 1 ] -=
			dt * flux[ edge->label - 1 ] * edge->length / edge->leftCell->area;
		if ( edge->rightCell )
			polution[ edge->rightCell->label - 1 ] +=
				dt * flux[ edge->label - 1 ] * edge->length / edge->rightCell->area;
	}
}    

/*
	Reads the parameters file.
*/
Parameters read_parameters (
	string parameter_filename)
{
	Parameters data;
	Parameter para( parameter_filename.c_str() );

	data.filenames.mesh = para.getString("MeshName");
	data.filenames.velocity = para.getString("VelocityFile");
	data.filenames.polution.initial = para.getString("PoluInitFile");
	data.filenames.polution.output = para.getString("PoluFile");
	data.time.final = para.getDouble("FinalTime");
	data.iterations.jump = para.getInteger("NbJump");
	data.computation.threshold = para.getDouble("DirichletCondition");

	return data;
}

/*
	Computes the mesh parameter (whatever that is)
*/
double compute_mesh_parameter (
	FVMesh2D mesh)
{
	double h;
	double S;
	FVCell2D *cell;
	FVEdge2D *edge;

	h = 1.e20;
	for ( mesh.beginCell(); ( cell = mesh.nextCell() ) ; )
	{
		S = cell->area;
		for ( cell->beginEdge(); ( edge = cell->nextEdge() ) ; )
		{
			//edge = cell->getEdge(e);
			if ( h * edge->length > S )
				h = S / edge->length;
		}
	}
	return h;
}

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
void main_loop (
	double final_time,						//	time computation limit
	unsigned jump_interval,					//	iterations output interval
	FVMesh2D mesh,							//	2D mesh to compute
	double mesh_parameter,					//	mesh parameter
	FVVect<double> polutions,				//	polution values vector
	FVVect<FVPoint2D<double> > velocities,	//	velocity vectors collection
	FVVect<double> fluxes,					//	flux values vector
	double dc,								//	Dirichlet condition
	string output_filename)					//	output file name
{
	double t;								//	time elapsed
	double dt;
	int i;									//	current iteration
	FVio polution_file( output_filename.c_str() ,FVWRITE);

	t = 0;
	i = 0;
	while ( t < final_time )
	{
		dt = compute_flux( mesh , polutions , velocities , fluxes , dc ) * mesh_parameter;
		update( mesh , polutions , fluxes , dt );
		t += dt;
		++i;
		if ( i % jump_interval == 0 )
		{
//			printf("step %d  at time %f \r", i, t);
//			fflush(NULL);
		}
	}
	polution_file.put( polutions , t , "polution" ); 
}

/*
	Função Madre
*/
int main(int argc, char** argv)
{  
	string name;
	double h;
	double t;
	FVMesh2D mesh;
	Parameters data;
#ifdef	PAPI_MEASURE_TIME
												//	last minute
	double cf_avg_ns;
#endif//	PAPI_MEASURE_TIME

#ifdef	PAPI_MEASURE
	//	init PAPI
	PAPI::init_threads();
#endif//	PAPI_MEASURE

	// read the parameter file
	if (argc > 1)
		data = read_parameters( string(argv[1]).c_str() );
	else
		data = read_parameters( "param.xml" );
	
	// read the mesh
	mesh.read( data.filenames.mesh.c_str() );

	FVVect<double> polution( mesh.getNbCell() );
	FVVect<double> flux( mesh.getNbEdge() );
	FVVect<FVPoint2D<double> > velocity( mesh.getNbCell() );

	//	read velocity
	FVio velocity_file( data.filenames.velocity.c_str() , FVREAD );
	velocity_file.get( velocity , t , name );

	//	read polution
	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVREAD );
	polu_ini_file.get( polution , t , name );

	//	OpenMP init
	tc = omp_get_num_procs() * OMP_FCT_CF;

	//	prepare velocities array
	//vs = new double[ mesh.getNbEdge() ];
	max_vel_v = new double[ tc ];
#ifdef	PAPI_MEASURE
#if defined		(PAPI_MEASURE_CPI)
	tot_cyc_v = new long long int[ tc ];
	tot_ins_v = new long long int[ tc ];
#elif defined		(PAPI_MEASURE_MEMORY)
	ld_ins_v = new long long int[ tc ];
	sr_ins_v = new long long int[ tc ];
#elif defined	(PAPI_MEASURE_FLOPS)
	tot_cyc_v = new long long int[ tc ];
	fp_ops_v = new long long int[ tc ];
#elif defined	(PAPI_MEASURE_L1)
	l1_dca_v = new long long int[ tc ];
	l1_dcm_v = new long long int[ tc ];
#elif defined	(PAPI_MEASURE_L2DCA)
	l2_dca_v = new long long int[ tc ];
#elif defined	(PAPI_MEASURE_IPB)
	tot_ins_v = new long long int[ tc ];
	l2_dcm_v = new long long int[ tc ];
#elif defined	(PAPI_MEASURE_MBAML)
	fp_ins_v = new long long int[ tc ];
	fml_ins_v = new long long int[ tc ];
#elif defined	(PAPI_MEASURE_MBADV)
	fp_ins_v = new long long int[ tc ];
	fdv_ins_v = new long long int[ tc ];
#endif
	tot_ns_v = new long long int[ tc ];
	
#if defined		(PAPI_MEASURE_CPI)
	tot_cyc = 0;
	tot_ins = 0;
#elif defined	(PAPI_MEASURE_MEMORY)
	ld_ins = 0;
	sr_ins = 0;
#elif defined	(PAPI_MEASURE_FLOPS)
	tot_cyc = 0;
	fp_ops = 0;
#elif defined	(PAPI_MEASURE_L1)
	l1_dca = 0;
	l1_dcm = 0;
#elif defined	(PAPI_MEASURE_L2DCA)
	l2_dca = 0;
#elif defined	(PAPI_MEASURE_IPB)
	tot_ins = 0;
	l2_dcm = 0;
#elif defined	(PAPI_MEASURE_MBAML)
	fp_ins = 0;
	fml_ins = 0;
#elif defined	(PAPI_MEASURE_MBADV)
	fp_ins = 0;
	fdv_ins = 0;
#endif
	tot_ns = 0;
#endif//	PAPI_MEASURE

#ifdef	PAPI_MEASURE_TIME
											//	compute_flux
	cf_tot_ns = 0;
	cf_max_ns = numeric_limits<unsigned long long int>::min();
	cf_min_ns = numeric_limits<unsigned long long int>::max();
	cf_count = 0;
#endif//	PAPI_MEASURE_TIME

	// compute the Mesh parameter
	h = compute_mesh_parameter( mesh );

	// the main loop
	main_loop(
		data.time.final,
		data.iterations.jump,
		mesh,
		h,
		polution,
		velocity,
		flux,
		data.computation.threshold,
		data.filenames.polution.output)
	;

#ifdef	PAPI_MEASURE_TIME
	//	last minute results
	cf_avg_ns = (double) cf_tot_ns / (double) cf_count;
#endif//	PAPI_MEASURE_TIME

	//	print PAPI results
	cout
		<<	"running:"	<<	argv[0]	<<	endl
#ifdef	PAPI_MEASURE
#if defined		(PAPI_MEASURE_CPI)
		<<	"totcyc:"	<<	tot_cyc	<<	endl
		<<	"totins:"	<<	tot_ins	<<	endl
#elif defined		(PAPI_MEASURE_MEMORY)
		<<	"ldins:"	<<	ld_ins	<<	endl
		<<	"srins:"	<<	sr_ins	<<	endl
#elif defined	(PAPI_MEASURE_FLOPS)
		<<	"totcyc:"	<<	tot_cyc	<<	endl
		<<	"fpops:"	<<	fp_ops	<<	endl
#elif defined	(PAPI_MEASURE_L1)
		<<	"l1dca:"	<<	l1_dca	<<	endl
		<<	"l1dcm:"	<<	l1_dcm	<<	endl
#elif defined	(PAPI_MEASURE_L2DCA)
		<<	"l2dca:"	<<	l2_dca	<<	endl
#elif defined	(PAPI_MEASURE_IPB)
		<<	"totins:"	<<	tot_ins	<<	endl
		<<	"l2dcm:"	<<	l2_dcm	<<	endl
#elif defined	(PAPI_MEASURE_MBAML)
		<<	"fpins:"	<<	fp_ins	<<	endl
		<<	"fmlins:"	<<	fml_ins	<<	endl
#elif defined	(PAPI_MEASURE_MBADV)
		<<	"fpins:"	<<	fp_ins	<<	endl
		<<	"fdvins:"	<<	fdv_ins	<<	endl
#endif
		<<	"totns:"	<<	tot_ns	<<	endl
#endif//	PAPI_MEASURE
#ifdef	PAPI_MEASURE_TIME
		<<	"cftotns:"	<<	cf_tot_ns	<<	endl
		<<	"cfmaxns:"	<<	cf_max_ns	<<	endl
		<<	"cfminns:"	<<	cf_min_ns	<<	endl
		<<	"cfavgns:"	<<	cf_avg_ns	<<	endl
#endif//	PAPI_MEASURE_TIME
	;

	//	cleanup
	delete max_vel_v;
#ifdef	PAPI_MEASURE
#if defined		(PAPI_MEASURE_CPI)
	delete tot_cyc_v;
	delete tot_ins_v;
#elif defined		(PAPI_MEASURE_MEMORY)
	delete ld_ins_v;
	delete sr_ins_v;
#elif defined	(PAPI_MEASURE_FLOPS)
	delete tot_cyc_v;
	delete fp_ops_v;
#elif defined	(PAPI_MEASURE_L1)
	delete l1_dca_v;
	delete l1_dcm_v;
#elif defined	(PAPI_MEASURE_L2DCA)
	delete l2_dca_v;
#elif defined	(PAPI_MEASURE_IPB)
	delete tot_ins_v;
	delete l2_dcm_v;
#elif defined	(PAPI_MEASURE_MBAML)
	delete fp_ins_v;
	delete fml_ins_v;
#elif defined	(PAPI_MEASURE_MBADV)
	delete fp_ins_v;
	delete fdv_ins_v;
#endif
	delete tot_ns_v;

	PAPI::shutdown();
#endif//	PAPI_MEASURE

	return 0;
}
