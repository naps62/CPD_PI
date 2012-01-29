#include <iostream>
#include <limits>

#include "FVLib.h"

//	BEGIN CONSTANTS

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
#define PAPI_MEASURE_SET
#define PAPI_MEASURE
#endif//	PAPI_MEASURE_ANY

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

#ifdef	PAPI_MEASURE
#ifdef	PAPI_MEASURE_SET
#if defined		(PAPI_MEASURE_CPI)
long long int tot_cyc;		//	process total cycles
long long int tot_ins;		//	process total instructions
#elif defined	(PAPI_MEASURE_MEMORY)
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
long long int fdv_ins;		//	process total FP divisions
#endif//	PAPI_MEASURE_ANY
#endif//	PAPI_MEASURE_SET

							//	compute_flux
unsigned
long long int cf_tot_ns;	//	compute_flux total useful time (nano seconds)
unsigned
long long int cf_max_ns;	//	compute_flux maximum useful time
unsigned
long long int cf_min_ns;	//	compute_flux minimum useful time
unsigned
long int cf_count;			//	compute_flux call count
#endif//	PAPI_MEASURE

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
	double dt;
	double p_left;							//	polution in the left face
	double p_right;							//	polution in the right face
	int i_left;								//	index of the left face
	int i_right;							//	index of the right face
	unsigned e;								//	edge iteration variable
	unsigned es;							//	total number of edges
	FVPoint2D<double> v_left;				//	velocity in the left face
	FVPoint2D<double> v_right;				//	velocity in the right face
	double v;								//	resulting velocity
	double v_max;							//	maximum resulting velocity
	FVEdge2D *edge;							//	current edge

											//	PAPI specific measurer
#ifdef	PAPI_MEASURE
#ifdef	PAPI_MEASURE_SET
#if defined		(PAPI_MEASURE_CPI)
	PAPI_CPI p;
#elif defined	(PAPI_MEASURE_MEMORY)
	PAPI_Memory p;
#elif defined	(PAPI_MEASURE_FLOPS)
	PAPI_Flops p;
#elif defined	(PAPI_MEASURE_L1)
	PAPI_L1 p;
#elif defined	(PAPI_MEASURE_L2DCA)
	PAPI_Custom p;
	p.add_event( PAPI_L2_DCA );
#elif defined	(PAPI_MEASURE_IPB)
	PAPI_InstPerByte p;
#elif defined	(PAPI_MEASURE_MBAML)
	PAPI_Custom p;
	p.add_event( PAPI_FP_INS );
	p.add_event( PAPI_FML_INS );
#elif defined	(PAPI_MEASURE_MBADV)
	PAPI_Custom p;
	p.add_event( PAPI_FP_INS );
	p.add_event( PAPI_FDV_INS );
#endif//	PAPI_MEASURE_ANY

	p.start();

#else//	no counter set
	
	unsigned
	long long int cf_ns;					//	current call useful time
	PAPI_Stopwatch sw;						//	a proper time measurer

	sw.start();

#endif//	PAPI_MEASURE_SET
#endif//	PAPI_MEASURE

	v_max = numeric_limits<double>::min();
	es = mesh.getNbEdge();
	for ( e = 0; e < es; ++e)
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
		v_max = ( v > v_max ) ? v : v_max;
		if ( v < 0 )
			flux[ edge->label - 1 ] = v * p_right;
		else
			flux[ edge->label - 1 ] = v * p_left;
	}

	dt = 1.0 / abs( v_max );

											//	retrieve PAPI values
#ifdef	PAPI_MEASURE
#ifdef	PAPI_MEASURE_SET

	p.stop();

#if defined		(PAPI_MEASURE_CPI)
	tot_cyc	+=	p.cycles();
	tot_ins	+=	p.instructions();
#elif defined	(PAPI_MEASURE_MEMORY)
	ld_ins	+=	p.loads();
	sr_ins	+=	p.stores();
#elif defined	(PAPI_MEASURE_FLOPS)
	tot_cyc	+=	p.cycles();
	fp_ops	+=	p.flops();
#elif defined	(PAPI_MEASURE_L1)
	l1_dca	+=	p.accesses();
	l1_dcm	+=	p.misses();
#elif defined	(PAPI_MEASURE_L2DCA)
	l2_dca	+=	p.get( PAPI_L2_DCA );
#elif defined	(PAPI_MEASURE_IPB)
	tot_ins	+=	p.instructions();
	l2_dcm	+=	p.ram_accesses();
#elif defined	(PAPI_MEASURE_MBAML)
	fp_ins	+=	p.get( PAPI_FP_INS );
	fml_ins	+=	p.get( PAPI_FML_INS );
#elif defined	(PAPI_MEASURE_MBADV)
	fp_ins	+=	p.get( PAPI_FP_INS );
	fdv_ins	+=	p.get( PAPI_FDV_INS );
#endif//	PAPI_MEASURE_ANY

	cf_ns	=	p.last_time();

#else//		PAPI_MEASURE_SET

	sw.stop();

	cf_ns	=	sw.total();

#endif//	PAPI_MEASURE_SET

	cf_tot_ns += cf_ns;
	cf_max_ns = ( cf_ns > cf_max_ns ) ? cf_ns : cf_max_ns;
	cf_min_ns = ( cf_ns < cf_min_ns ) ? cf_ns : cf_min_ns;
	cf_count += 1;

#endif//	PAPI_MEASURE

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
//	unsigned jump_interval,					//	iterations output interval
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
//	polution_file.put( polutions , t , "polution" ); 
	while ( t < final_time )
	{
		dt = compute_flux( mesh , polutions , velocities , fluxes , dc ) * mesh_parameter;
		update( mesh , polutions , fluxes , dt );
		t += dt;
		++i;
//		if ( i % jump_interval == 0 )
//			polution_file.put( polutions , t , "polution" );    
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
#ifdef	PAPI_MEASURE
											//	last minute
	double cf_avg_ns;

	//	init PAPI
	PAPI::init();

#endif//	PAPI_MEASURE

	// read the parameter
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

	//	prepare variables
#ifdef	PAPI_MEASURE
#ifdef	PAPI_MEASURE_SET
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
#endif//	PAPI_MEASURE_ANY
#endif//	PAPI_MEASURE_SET

	cf_tot_ns = 0;
	cf_max_ns = numeric_limits<unsigned long long int>::min();
	cf_min_ns = numeric_limits<unsigned long long int>::max();
	cf_count = 0;

#endif//	PAPI_MEASURE

	// compute the Mesh parameter
	h = compute_mesh_parameter( mesh );

	// the main loop
	main_loop(
		data.time.final,
//		data.iterations.jump,
		mesh,
		h,
		polution,
		velocity,
		flux,
		data.computation.threshold,
		data.filenames.polution.output)
	;

#ifdef	PAPI_MEASURE
											//	last minute
	cf_avg_ns = (double) cf_tot_ns / (double) cf_count;

	tot_ns = cf_tot_ns;						//	process total useful measured time

#endif//	PAPI_MEASURE

	cout
		<<	"running:"	<<	argv[0]	<<	endl
#ifdef	PAPI_MEASURE
#ifdef	PAPI_MEASURE_SET
#if defined		(PAPI_MEASURE_CPI)
		<<	"totcyc:"	<<	tot_cyc	<<	endl
		<<	"totins:"	<<	tot_ins	<<	endl
#elif defined	(PAPI_MEASURE_MEMORY)
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
#endif//	PAPI_MEASURE_ANY
#endif//	PAPI_MEASURE_SET

		<<	"cftotns:"	<<	cf_tot_ns	<<	endl
		<<	"cfmaxns:"	<<	cf_max_ns	<<	endl
		<<	"cfminns:"	<<	cf_min_ns	<<	endl
		<<	"cfavgns:"	<<	cf_avg_ns	<<	endl

		<<	"totns:"	<<	tot_ns	<<	endl

#endif//	PAPI_MEASURE
	;
}
