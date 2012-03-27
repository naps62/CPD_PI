#include <iostream>
#include <limits>
#include "FVLib.h"

#include <fv/cpu/cell.hpp>
using fv::cpu::Cell;
#include <fv/cpu/edge.hpp>
using fv::cpu::Edge;




#if   defined (PROFILE_BTM) \
 ||   defined (PROFILE_L1DCM) \
 ||   defined (PROFILE_L2TCM) \
 ||   defined (PROFILE_L2DCM)
#define PROFILE_MEMORY
#endif

#if   defined (PROFILE_TOTINS)	\
 ||   defined (PROFILE_LDINS) \
 ||   defined (PROFILE_SRINS) \
 ||   defined (PROFILE_BRINS) \
 ||   defined (PROFILE_FPINS) \
 ||   defined (PROFILE_VECINS)
#define PROFILE_INSTRUCTIONS
#endif

#if   defined (PROFILE_FLOPS)
#define PROFILE_OPERATIONS
#endif

#if   defined (PROFILE_MEMORY) \
 ||   defined (PROFILE_INSTRUCTIONS) \
 ||   defined (PROFILE_OPERATIONS)
#define PROFILE
#endif





//
//  GLOBAL
//
#if   defined (PROFILE)
#include <papi/papi.hpp>
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
#include <papi/memacs.hpp>
papi::TotalMemoryAccessesCounter *p;
//long long int btm;
#elif defined (PROFILE_L1DCM)
#include <papi/l1dcm.hpp>
papi::L1DataCacheMissesPresetCounter *p;
long long int l1dcm;
#elif defined (PROFILE_L2TCM)
#include <papi/l2tcm.hpp>
papi::L2TotalCacheMissesPresetCounter *p;
long long int l2tcm;
#elif defined (PROFILE_L2DCM)
#include <papi/l2dcm.hpp>
papi::L2DataCacheMissesPresetCounter *p;
long long int l2dcm;
#endif//	PROFILE_*
#endif//	PROFILE_MEMORY
#if   defined (PROFILE_INSTRUCTIONS)
#include <papi/papiinst.hpp>
#if   defined (PROFILE_BRINS)
#include <papi/brins.hpp>
papi::BranchInstructionsPresetCounter *p;
long long int brins;
#elif defined (PROFILE_FPINS)
#include <papi/fpins.hpp>
papi::FloatingPointInstructionsPresetCounter *p;
long long int fpins;
#elif defined (PROFILE_LDINS)
#include <papi/ldins.hpp>
papi::LoadInstructionsPresetCounter *p;
long long int ldins;
#elif defined (PROFILE_SRINS)
#include <papi/srins.hpp>
papi::StoreInstructionsPresetCounter *p;
long long int srins;
#elif defined (PROFILE_TOTINS)
//#include <papi/totins.hpp>
papi::TotalInstructionsCounter *p;
//long long int totins;
#elif defined (PROFILE_VECINS)
#include <papi/vecins.hpp>
papi::VectorInstructionsPresetCounter *p;
long long int vecins;
#endif//PROFILE_*
#endif//PROFILE_INSTRUCTIONS
#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
#include <papi/flops.hpp>
papi::FloatingPointOperationsCounter *p;
long long int flops;
#endif//PROFILE_FLOPS
#endif//PROFILE_OPERATIONS
long long int mlbegin;
long long int cftotns;
long long int cfminns;
long long int cfmaxns;
long long int uptotns;
long long int upminns;
long long int upmaxns;
long long int ctl;
unsigned int mliters;
#endif//PROFILE





void compute_flux(
	Edge *edges,
	unsigned edge_count,
	Cell *cells,
	double dirichlet)
{
	double polution_left;
	double polution_right;

#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
#endif//PROFILE_WARMUP
	p->start();
#endif//PROFILE

	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[e];
		Cell &cell_left = cells[ edge.left ];
		polution_left = cell_left.polution;
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
			Cell &cell_right = cells[ edge.right ];
			polution_right = cell_right.polution;
		}
		else
		{
			polution_right= dirichlet;
		} 
		edge.flux = ( edge.velocity < 0 )
				  ? ( edge.velocity * polution_right )
				  : ( edge.velocity * polution_left );
	}

#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
	{
#endif//PROFILE_WARMUP
	p->stop();
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
//	btm += p->transactions() - ctl;
#elif defined (PROFILE_L1DCM)
	l1dcm += p->misses() - ctl;
#elif defined (PROFILE_L2TCM)
	l2tcm += p->misses() - ctl;
#elif defined (PROFILE_L2DCM)
	l2dcm += p->misses() - ctl;
#endif//PROFILE_*
#endif//PROFILE_MEMORY
#if   defined (PROFILE_INSTRUCTIONS)
	//long long int inst = p->instructions() - ctl;
#if   defined (PROFILE_BRINS)
	brins += inst;
#elif defined (PROFILE_FPINS)
	fpins += inst;
#elif defined (PROFILE_LDINS)
	ldins += inst;
#elif defined (PROFILE_SRINS)
	srins += inst;
#elif defined (PROFILE_TOTINS)
//	totins += inst;
#elif defined (PROFILE_VECINS)
	vecins += inst;
#endif//PROFILE_*
#endif//PROFILE_INSTRUCTIONS
#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
	flops += p->operations() - ctl;
#endif//PROFILE_FLOPS
#endif//PROFILE_OPERATIONS
	{
		long long int timens = p->last();
		cftotns += timens;
		cfmaxns = ( timens > cfmaxns ) ? timens : cfmaxns;
		cfminns = ( timens < cfminns ) ? timens : cfminns;
	}
#if   defined (PROFILE_WARMUP)
	}
#endif//PROFILE_WARMUP
#endif//PROFILE
}






void    update(
	Cell *cells,
	unsigned cell_count,
	Edge *edges,
	double dt)
{

#if   defined PROFILE
#if   defined PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
#endif//PROFILE_WARMUP
	p->start();
#endif//PROFILE

	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		Cell &cell = cells[ c ];
		double cdp = 0;

		for ( unsigned e = 0 ; e < cell.edge_count ; ++e )
		{
			Edge &edge = edges[ cell.edges[ e ] ];
			double edp = dt * edge.flux * edge.length / cell.area;
			if ( c == edge.left )
				cdp -= edp;
			else
				cdp += edp;
		}

		cell.polution += cdp;
	}

#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
	{
#endif//PROFILE_WARMUP
	p->stop();
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
//	btm += p->transactions() - ctl;
#elif defined (PROFILE_L1DCM)
	l1dcm += p->misses() - ctl;
#elif defined (PROFILE_L2TCM)
	l2tcm += p->misses() - ctl;
#elif defined (PROFILE_L2DCM)
	l2dcm += p->misses() - ctl;
#endif//PROFILE_*
#endif//PROFILE_MEMORY
#if   defined (PROFILE_INSTRUCTIONS)
//	long long int inst = p->instructions() - ctl_ins;
#if   defined (PROFILE_BRINS)
	br_ins += inst;
#elif defined (PROFILE_FPINS)
	fp_ins += inst;
#elif defined (PROFILE_LDINS)
	ld_ins += inst;
#elif defined (PROFILE_SRINS)
	sr_ins += inst;
#elif defined (PROFILE_TOTINS)
	//tot_ins += inst;
#elif defined (PROFILE_VECINS)
	int_ins += inst;
#endif//PROFILE_*
#endif//PROFILE_INSTRUCTIONS
#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
	flops += p->operations()
#endif//PROFILE_FLOPS
#endif//PROFILE_OPERATIONS
	{
		long long int timens = p->last();
		uptotns += timens;
		upmaxns = ( timens > upmaxns ) ? timens : upmaxns;
		upminns = ( timens < upminns ) ? timens : upminns;
	}
#if   defined (PROFILE_WARMUP)
	}
#endif//PROFILE_WARMUP
#endif//PROFILE

}    















int main(int argc, char *argv[])
{
#if   defined (PROFILE)
	papi::init();
	long long int totalns = papi::time::real::nanoseconds();
#endif//PROFILE
	string parameter_filename;
	
	if ( argc > 1 )
		parameter_filename = argv[1];
	else
		parameter_filename = "param.xml";

	string mesh_filename,velo_filename,pol_filename,pol_ini_filename;
	string name;
	// read the parameter
	Parameter para(parameter_filename.c_str());
	mesh_filename=para.getString("MeshName");
	velo_filename=para.getString("VelocityFile");
	pol_ini_filename=para.getString("PoluInitFile");
	string pol_fname = para.getString( "PoluFile" );

	double dirichlet = para.getDouble("DirichletCondition");

	double time,final_time,dt,h,S;
	size_t nbiter,nbjump;
	FVMesh2D m;
	FVCell2D *ptr_c;
	FVEdge2D *ptr_e;
	// read the mesh
	m.read(mesh_filename.c_str());
	FVVect<double> pol(m.getNbCell()),flux(m.getNbEdge());
	FVVect<FVPoint2D<double> > V(m.getNbCell());
	// read the  data  and initialisation
	FVio velocity_file(velo_filename.c_str(),FVREAD);
	velocity_file.get(V,time,name);
	FVio polu_ini_file(pol_ini_filename.c_str(),FVREAD);
	polu_ini_file.get(pol,time,name);
	final_time=para.getDouble("FinalTime");
	nbjump=para.getInteger("NbJump");
	// compute the Mesh parameter
	h=1.e20;
	m.beginCell();
	while((ptr_c=m.nextCell()))
	{
		S=ptr_c->area;
		ptr_c->beginEdge();
		while((ptr_e=ptr_c->nextEdge()))
		{
			if(h*ptr_e->length>S) h=S/ptr_e->length;
		}
	}


	unsigned edge_count = m.getNbEdge();
	double max_vel = numeric_limits<double>::min();
	Edge *edges = new Edge[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[ e ];
		FVEdge2D *fv_edge = m.getEdge( e );

		edge.flux = flux[ e ];
		edge.length = fv_edge->length;
//		edge.normal[0] = fv_edge->normal.x;
//		edge.normal[1] = fv_edge->normal.y;
		edge.left = fv_edge->leftCell->label - 1;
		edge.right = ( fv_edge->rightCell )
					 ? fv_edge->rightCell->label - 1
					 : numeric_limits<unsigned>::max();

		double normal[2];
		normal[0] = fv_edge->normal.x;
		normal[1] = fv_edge->normal.y;
		double v_left[2];
		v_left[0] = V[ edge.left ].x;
		v_left[1] = V[ edge.left ].y;
		double v_right[2];
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
			v_right[0] = V[ edge.right ].x;
			v_right[1] = V[ edge.right ].y;
		}
		else
		{
			v_right[0] = v_left[0];
			v_right[1] = v_left[1];
		}

		edge.velocity = ( v_left[0] + v_right[0] ) * 0.5 * normal[0]
					  + ( v_left[1] + v_right[1] ) * 0.5 * normal[1];

		max_vel = ( abs( edge.velocity ) > max_vel )
				? abs( edge.velocity )
				: max_vel
				;
	}


	dt = h / max_vel;



	unsigned cell_count = m.getNbCell();
	Cell *cells = new Cell[ cell_count ];
	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		Cell &cell = cells[ c ];
		FVCell2D *fv_cell = m.getCell( c );

		cell.velocity[0] = V[ c ].x;
		cell.velocity[1] = V[ c ].y;
		cell.polution = pol[ c ];
		cell.area = fv_cell->area;
		cell.init( fv_cell->nb_edge );
		for ( unsigned e = 0 ; e < cell.edge_count ; ++e )
			cell.edges[ e ] = fv_cell->edge[ e ]->label - 1;
	}



#if   defined (PROFILE)
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
	p = new papi::TotalMemoryAccessesCounter();
//	btm = 0;
	p->start();
	p->stop();
//	ctl = p->transactions();
#elif defined (PROFILE_L1DCM)
	p = new papi::L1DataCacheMissesPresetCounter();
	l1dcm = 0;
	p->start();
	p->stop();
	ctl = p->misses();
#elif defined (PROFILE_L2TCM)
	p = new papi::L2TotalCacheMissesPresetCounter();
	l2tcm = 0;
	p->start();
	p->stop();
	ctl = p->misses();
#elif defined (PROFILE_L2DCM)
	p = new papi::L2DataCacheMissesPresetCounter();
	l2dcm = 0;
	p->start();
	p->stop();
	ctl = p->misses();
#endif//PROFILE_*
#endif//PROFILE_MEMORY
#if   defined (PROFILE_INSTRUCTIONS)
#if   defined (PROFILE_BRINS)
	p = new papi::BranchInstructionsPresetCounter();
	brins = 0;
#elif defined (PROFILE_FPINS)
	p = new papi::FloatingPointInstructionsPresetCounter();
	fpins = 0;
#elif defined (PROFILE_LDINS)
	p = new papi::LoadInstructionsPresetCounter();
	ldins = 0;
#elif defined (PROFILE_SRINS)
	p = new papi::StoreInstructionsPresetCounter();
	srins = 0;
#elif defined (PROFILE_TOTINS)
	p = new papi::TotalInstructionsCounter();
	//totins = 0;
#elif defined (PROFILE_VECINS)
	p = new papi::VectorInstructionsPresetCounter();
	vecins = 0;
#endif//PROFILE_*
	//p->start();
	//p->stop();
	//ctl = p->instructions_l();
#endif//PROFILE_INSTRUCTIONS
#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
	p = new papi::FloatingPointInstructionsPresetCounter();
	flops = 0;
#endif//PROFILE_FLOPS
#endif//PROFILE_OPERATIONS
	long long int mltotns = 0;
	long long int mlmaxns = numeric_limits<long long int>::min();
	long long int mlminns = numeric_limits<long long int>::max();
	cftotns = 0;
	cfmaxns = numeric_limits<long long int>::min();
	cfminns = numeric_limits<long long int>::max();
	uptotns = 0;
	upmaxns = numeric_limits<long long int>::min();
	upminns = numeric_limits<long long int>::max();
	mliters = 0;
#endif//PROFILE




	// the main loop
	time=0.;nbiter=0;
	FVio pol_file( pol_fname.c_str() ,FVWRITE);
#if   defined (PROFILE)
#if   defined (PROFILE_LIMITED)
	for ( int i = 0 ; i < PROFILE_LIMITED ; ++i )
#else//PROFILE_LIMITED
	while ( time < final_time )
#endif//PROFILE_LIMITED
	{
#if   defined (PROFILE_WARMUP)
		if ( mliters > PROFILE_WARMUP )
#endif//PROFILE_WARMUP
			mlbegin = papi::time::real::nanoseconds();
#else//PROFILE
	while(time<final_time)
	{
#endif//PROFILE
		compute_flux(
			edges,
			edge_count,
			cells,
			dirichlet)
		;

		update(
			cells,
			cell_count,
			edges,
			dt);
		time += dt;
#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
		if ( mliters > PROFILE_WARMUP )
		{
#endif//PROFILE_WARMUP
			long long int timens = papi::time::real::nanoseconds() - mlbegin;
			mltotns += timens;
			mlmaxns = ( timens > mlmaxns ) ? timens : mlmaxns;
			mlminns = ( timens < mlminns ) ? timens : mlminns;
#if   defined (PROFILE_WARMUP)
		}
#endif//PROFILE_WARMUP
		++mliters;
#endif//PROFILE
	}

	for ( unsigned c = 0; c < cell_count ; ++c )
		pol[ c ] = cells[ c ].polution;

	pol_file.put(pol,time,"polution"); 

	delete[] cells;
	delete[] edges;

#if   defined (PROFILE)
	delete p;
	papi::shutdown();
	totalns = papi::time::real::nanoseconds() - totalns;
	cout
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
				//<<	btm
				<<	p->accesses_t()
#elif defined (PROFILE_L1DCM)
				<<	l1dcm
#elif defined (PROFILE_L2TCM)
				<<	l2tcm
#elif defined (PROFILE_L2DCM)
				<<	l2dcm
#endif//PROFILE_*
#endif//PROFILE_MEMORY
#if   defined (PROFILE_INSTRUCTIONS)
#if   defined (PROFILE_BRINS)
				<<	brins
#elif defined (PROFILE_FPINS)
				<<	fpins
#elif defined (PROFILE_LDINS)
				<<	ldins
#elif defined (PROFILE_SRINS)
				<<	srins
#elif defined (PROFILE_TOTINS)
				//<<	totins
				<<	p->instructions_t()
#elif defined (PROFILE_VECINS)
				<<	vecins
#endif//PROFILE_*
#endif//PROFILE_INSTRUCTIONS
#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
				<<	flops
#endif//PROFILE_FLOPS
#endif//PROFILE_OPERATIONS
		<<	';'	<<	totalns
		<<	';'	<<	mltotns
		<<	';'	<<	mlmaxns
		<<	';'	<<	mlminns
		<<	';'	<<	cftotns
		<<	';'	<<	cfmaxns
		<<	';'	<<	cfminns
		<<	';'	<<	uptotns
		<<	';'	<<	upmaxns
		<<	';'	<<	upminns
						<<	endl
		;
#endif//PROFILE
}
