#include <iostream>
#include <limits>
#include "FVLib.h"



#if   defined (PROFILE_BTM) \
 ||   defined (PROFILE_L1DCM) \
 ||   defined (PROFILE_L2DCM)
#define PROFILE_MEMORY
#endif

#if   defined (PROFILE_TOTINS) \
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
//	GLOBALS
//
#if   defined (PROFILE)

#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
#include <papi/bytes_accessed.hpp>
papi::BytesAccessed *p;
long long int btm_bytes;
#elif   defined (PROFILE_L1DCM)
#include <papi/l1dcm.hpp>
papi::L1DataCacheMissesCounter *p;
long long int l1dcm;
#elif   defined (PROFILE_L2DCM)
#include <papi/l2dcm.hpp>
papi::L2DataCacheMissesCounter *p;
long long int l2dcm;
#endif//    PROFILE_*
#endif//    PROFILE_MEMORY

#if   defined (PROFILE_INSTRUCTIONS)
#if   defined (PROFILE_BRINS)
#include <papi/brins.hpp>
papi::BranchInstructionsCounter *p;
long long int brins;
#elif defined (PROFILE_FPINS)
#include <papi/fpins.hpp>
papi::FloatingPointInstructionsCounter *p;
long long int fpins;
#elif defined (PROFILE_LDINS)
#include <papi/ldins.hpp>
papi::LoadInstructionsCounter *p;
long long int ldins;
#elif defined (PROFILE_SRINS)
#include <papi/srins.hpp>
papi::StoreInstructionsCounter *p;
long long int srins;
#elif defined (PROFILE_TOTINS)
#include <papi/totins.hpp>
papi::TotalInstructionsCounter *p;
long long int totins;
#elif defined (PROFILE_VECINS)
#include <papi/vecins.hpp>
papi::VectorInstructionsCounter *p;
long long int vecins;
#endif//	PROFILE_*
#endif//	PROFILE_INSTRUCTIONS

#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
#include <papi/flops.hpp>
papi::FloatingPointOperationsCounter *p;
long long int flops;
#endif//	PROFILE_FLOPS
#endif//	PROFILE_OPERATIONS

long long int cftotns;
long long int cfminns;
long long int cfmaxns;

long long int uptotns;
long long int upminns;
long long int upmaxns;

#if   defined (PROFILE_WARMUP)
unsigned mliters;
#endif//	PROFILE_WARMUP

#endif//    PROFILE



void
compute_flux
	(
	double *   polutions,
	double *   velocities,
	unsigned * lefts,
	unsigned * rights,
	double *   fluxes,
	double     dirichlet,
	unsigned   edge_count
	)
{
#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
#endif//	PROFILE_WARMUP
	p->start();
#endif//    PROFILE
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		double polution_left = polutions[ lefts[e] ];
		double polution_right
			= ( rights[e] < numeric_limits<unsigned>::max() )
			? polutions[ rights[e] ]
			: dirichlet
			;
		fluxes[e] = ( velocities[e] < 0 )
		          ? velocities[e] * polution_right
				  : velocities[e] * polution_left
				  ;
	}
#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
	{
#endif//	PROFILE_WARMUP
	p->stop();
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
	btm_bytes += p->bytes();
#elif defined (PROFILE_L1DCM)
	l1dcm += p->misses();
#elif defined (PROFILE_L2DCM)
	l2dcm += p->misses();
#endif//    PROFILE_*
#endif//    PROFILE_MEMORY

#if   defined (PROFILE_INSTRUCTIONS)
#if   defined (PROFILE_BRINS)
	brins += p->instructions();
#elif defined (PROFILE_FPINS)
	fpins += p->instructions();
#elif defined (PROFILE_LDINS)
	ldins += p->instructions();
#elif defined (PROFILE_SRINS)
	srins += p->instructions();
#elif defined (PROFILE_TOTINS)
	totins += p->instructions();
#elif defined (PROFILE_VECINS)
	vecins += p->instructions();
#endif//	PROFILE_*
#endif//	PROFILE_INSTRUCTIONS

#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
	flops += p->operations();
#endif//	PROFILE_FLOPS
#endif//	PROFILE_OPERATIONS
	{
		long long int timens = p->last_time();
		cftotns += timens;
		cfmaxns = ( timens > cfmaxns ) ? timens : cfmaxns;
		cfminns = ( timens < cfminns ) ? timens : cfminns;
	}
#if   defined (PROFILE_WARMUP)
	}
#endif//	PROFILE_WARMUP
#endif//    PROFILE
}






void
update
	(
	double *   polutions,
	double *   areas,
	double *   fluxes,
	double *   lengths,
	unsigned * indexes,
	unsigned * edges,
	unsigned * lefts,
	double     dt,
	unsigned   index_count,
	unsigned   cell_count
	)
{
#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
#endif
	p->start();
#endif//    PROFILE
	unsigned cell_last = cell_count - 1;

	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		double cdp = 0;
		unsigned i_limit
			= ( c < cell_last )
			? indexes[c+1]
			: index_count
			;
		for ( unsigned i = indexes[c] ; i < i_limit ; ++i )
		{
			unsigned e = edges[i];
			double edp = dt * fluxes[e] * lengths[e] / areas[c];
			if ( lefts[e] == c )
				cdp -= edp;
			else
				cdp += edp;
		}

		polutions[c] += cdp;
	}
#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
	if ( mliters > PROFILE_WARMUP )
	{
#endif
	p->stop();
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
	btm_bytes += p->bytes();
#elif defined (PROFILE_L1DCM)
	l1dcm += p->misses();
#elif defined (PROFILE_L2DCM)
	l2dcm += p->misses();
#endif//    PROFILE_*
#endif//    PROFILE_MEMORY

#if   defined (PROFILE_INSTRUCTIONS)
#if   defined (PROFILE_BRINS)
	brins += p->instructions();
#elif defined (PROFILE_FPINS)
	fpins += p->instructions();
#elif defined (PROFILE_LDINS)
	ldins += p->instructions();
#elif defined (PROFILE_SRINS)
	srins += p->instructions();
#elif defined (PROFILE_TOTINS)
	totins += p->instructions();
#elif defined (PROFILE_VECINS)
	vecins += p->instructions();
#endif//	PROFILE_*
#endif//	PROFILE_INSTRUCTIONS

#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
	flops += p->operations();
#endif//	PROFILE_FLOPS
#endif//	PROFILE_OPERATIONS
	{
		long long int timens = p->last_time();
		uptotns += timens;
		upmaxns = ( timens > upmaxns ) ? timens : upmaxns;
		upminns = ( timens < upminns ) ? timens : upminns;
	}
#if   defined (PROFILE_WARMUP)
	}
#endif
#endif//    PROFILE
}















int main(int argc, char *argv[])
{
#if   defined (PROFILE)
	papi::init();
	long long int totalns = papi::real_nano_seconds();
#endif//	PROFILE
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
	string pol_fname = para.getString("PoluFile");

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



#if   defined (PROFILE)
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
	p = new papi::BytesAccessed();
	btm_bytes = 0;
#elif defined (PROFILE_L1DCM)
	p = new papi::L1DataCacheMissesCounter();
	l1dcm = 0;
#elif defined (PROFILE_L2DCM)
	p = new papi::L2DataCacheMissesCounter();
	l2dcm = 0;
#endif//    PROFILE_*
#endif//    PROFILE_MEMORY

#if   defined (PROFILE_INSTRUCTIONS)
#if   defined (PROFILE_BRINS)
	p = new papi::BranchInstructionsCounter();
	brins = 0;
#elif defined (PROFILE_FPINS)
	p = new papi::FloatingPointInstructionsCounter();
	fpins = 0;
#elif defined (PROFILE_LDINS)
	p = new papi::LoadInstructionsCounter();
	ldins = 0;
#elif defined (PROFILE_SRINS)
	p = new papi::StoreInstructionsCounter();
	srins = 0;
#elif defined (PROFILE_TOTINS)
	p = new papi::TotalInstructionsCounter();
	totins = 0;
#elif defined (PROFILE_VECINS)
	p = new papi::VectorInstructionsCounter();
	vecins = 0;
#endif//	PROFILE_*
#endif//	PROFILE_INSTRUCTIONS

#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
	p = new papi::FloatingPointOperationsCounter();
	flops = 0;
#endif//	PROFILE_FLOPS
#endif//	PROFILE_OPERATIONS

	long long int mltotns = 0;
	long long int mlmaxns = numeric_limits<long long int>::min();
	long long int mlminns = numeric_limits<long long int>::max();
	
	cftotns = 0;
	cfmaxns = numeric_limits<long long int>::min();
	cfminns = numeric_limits<long long int>::max();

	uptotns = 0;
	upmaxns = numeric_limits<long long int>::min();
	upminns = numeric_limits<long long int>::max();

#endif//    PROFILE



	unsigned   edge_count = m.getNbEdge();
	double     max_vel    = numeric_limits<double>::min();
	double *   fluxes     = new double[ edge_count ];
	double *   lengths    = new double[ edge_count ];
	double *   velocities = new double[ edge_count ];
	unsigned * lefts      = new unsigned[ edge_count ];
	unsigned * rights     = new unsigned[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		FVEdge2D *fv_edge = m.getEdge( e );

		fluxes[ e ]   = flux[ e ];
		lengths[ e ]  = fv_edge->length;
		lefts[e]  = fv_edge->leftCell->label - 1;
		rights[e] = ( fv_edge->rightCell )
					  ? fv_edge->rightCell->label - 1
					  : numeric_limits<unsigned>::max();

		double normal[2];
		normal[0] = fv_edge->normal.x;
		normal[1] = fv_edge->normal.y;
		double v_left[2];
		v_left[0] = V[ lefts[e] ].x;
		v_left[1] = V[ lefts[e] ].y;
		double v_right[2];
		if ( rights[e] < numeric_limits<unsigned>::max() )
		{
			v_right[0] = V[ rights[e] ].x;
			v_right[1] = V[ rights[e] ].y;
		}
		else
		{
			v_right[0] = v_left[0];
			v_right[1] = v_left[1];
		}

		velocities[e] = ( v_left[0] + v_right[0] ) * 0.5 * normal[0]
					  + ( v_left[1] + v_right[1] ) * 0.5 * normal[1];

		max_vel = ( abs( velocities[e] ) > max_vel )
				? abs( velocities[e] )
				: max_vel
				;
	}


	dt = h / max_vel;



	unsigned   cell_count      = m.getNbCell();
	unsigned   cell_edge_count = 0;
	//Cell *cells = new Cell[ cell_count ];
	double *   polutions       = new double[ cell_count ];
	double *   areas           = new double[ cell_count ];
	unsigned * indexes         = new unsigned[ cell_count ];
	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		FVCell2D *fv_cell = m.getCell( c );

		polutions[c] = pol[c];
		areas[c] = fv_cell->area;
		indexes[c] = cell_edge_count;
		cell_edge_count += fv_cell->nb_edge;
	}




	unsigned   i         = 0;
	unsigned   c         = 0;
	unsigned   cell_last = cell_count - 1;
	unsigned * edges     = new unsigned[ cell_edge_count ];
	while ( c < cell_last )
	{
		FVCell2D *fv_cell = m.getCell( c );
		unsigned e = 0;
		while ( i < indexes[c+1] )
		{
			edges[i] = fv_cell->edge[e]->label - 1;
			++e;
			++i;
		}
		++c;
	}{
		FVCell2D *fv_cell = m.getCell( c );
		unsigned e = 0;
		while ( i < cell_edge_count )
		{
			edges[i] = fv_cell->edge[e]->label - 1;
			++e;
			++i;
		}
	}		



	// the main loop
	time=0.;
	FVio pol_file( pol_fname.c_str() ,FVWRITE);
	
#if   defined (PROFILE)
#if   defined (PROFILE_LIMITED)
	for ( int i = 0 ; i < PROFILE_LIMITED ; ++i )
#else//	PROFILE_LIMITED
	while( time < final_time)
#endif//	PROFILE_LIMITED
	{
#if   defined (PROFILE_WARMUP)
		if ( mliters > PROFILE_WARMUP )
#endif//	PROFILE_WARMUP
		long long int mlbegin = papi::real_nano_seconds();
#else//    PROFILE
	while( time < final_time)
	{
#endif//    PROFILE
		compute_flux(
			polutions,
			velocities,
			lefts,
			rights,
			fluxes,
			dirichlet,
			edge_count)
		;

		update(
			polutions,
			areas,
			fluxes,
			lengths,
			indexes,
			edges,
			lefts,
			dt,
			cell_edge_count,
			cell_count)
		;


		time += dt;
#if   defined (PROFILE)
#if   defined (PROFILE_WARMUP)
		if ( mliters > PROFILE_WARMUP )
		{
#endif//	PROFILE_WARMUP
		long long int timens = papi::real_nano_seconds() - mlbegin;
		mltotns += timens;
		mlmaxns = ( timens > mlmaxns ) ? timens : mlmaxns;
		mlminns = ( timens < mlminns ) ? timens : mlminns;
#if   defined (PROFILE_WARMUP)
		}
		mliters++;
#endif//	PROFILE_WARMUP
#endif//	PROFILE
	}

	for ( unsigned c = 0; c < cell_count ; ++c )
		pol[ c ] = polutions[c];

	pol_file.put(pol,time,"polution"); 

	delete[] fluxes;
	delete[] lengths;
	delete[] velocities;
	delete[] lefts;
	delete[] rights;
	delete[] polutions;
	delete[] areas;
	delete[] indexes;
	delete[] edges;


#if   defined (PROFILE)
	delete p;
	totalns = papi::real_nano_seconds() - totalns;
	cout
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
				<<	btm_bytes
#elif defined (PROFILE_L1DCM)
				<<	l1dcm
#elif defined (PROFILE_L2DCM)
				<<	l2dcm
#endif//    PROFILE_*
#endif//    PROFILE_MEMORY
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
				<<	totins
#elif defined (PROFILE_VECINS)
				<<	vecins
#endif//	PROFILE_*
#endif//	PROFILE_INSTRUCTIONS
#if   defined (PROFILE_OPERATIONS)
#if   defined (PROFILE_FLOPS)
				<<	flops
#endif//	PROFILE_FLOPS
#endif//	PROFILE_OPERATIONS
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
#endif//    PROFILE
}
