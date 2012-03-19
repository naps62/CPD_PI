#include <iostream>
#include <limits>
#include "FVLib.h"



#if   defined (PROFILE_L1DCM) \
 ||   defined (PROFILE_L2DCM) \
 ||   defined (PROFILE_BTM)
#define PROFILE_MEMORY
#endif

#if   defined (PROFILE_TOTINS) \
 ||   defined (PROFILE_LDINS) \
 ||   defined (PROFILE_SRINS) \
 ||   defined (PROFILE_BRINS) \
 ||   defined (PROFILE_FPINS)
#define PROFILE_INSTRUCTIONS
#endif

#if   defined (PROFILE_MEMORY) \
 ||   defined (PROFILE_INSTRUCTIONS)
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
#endif//    PROFILE_BTM
#endif//    PROFILE_MEMORY
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
	p->stop();
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
	btm_bytes += p->bytes();
#endif//    PROFILE_BTM
#endif//    PROFILE_MEMORY
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
	p->stop();
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
	btm_bytes += p->bytes();
#endif//    PROFILE_BTM
#endif//    PROFILE_MEMORY
#endif//    PROFILE
}















int main(int argc, char *argv[])
{  
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
#endif//    PROFILE_BTM
#endif//    PROFILE_MEMORY
#endif//    PROFILE



	unsigned edge_count = m.getNbEdge();
	double max_vel = numeric_limits<double>::min();
	//Edge *edges = new Edge[ edge_count ];
	double *   fluxes     = new double[ edge_count ];
	double *   lengths    = new double[ edge_count ];
	double *   velocities = new double[ edge_count ];
	unsigned * lefts  = new unsigned[ edge_count ];
	unsigned * rights = new unsigned[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		//Edge &edge = edges[ e ];
		FVEdge2D *fv_edge = m.getEdge( e );

		//edge.flux = flux[ e ];
		fluxes[ e ]   = flux[ e ];
		//edge.length = fv_edge->length;
		lengths[ e ]  = fv_edge->length;
		//edge.left = fv_edge->leftCell->label - 1;
		lefts[e]  = fv_edge->leftCell->label - 1;
		//edge.right = ( fv_edge->rightCell )
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
		//if ( edge.right < numeric_limits<unsigned>::max() )
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

		//edge.velocity = ( v_left[0] + v_right[0] ) * 0.5 * normal[0]
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
		//Cell &cell = cells[ c ];
		FVCell2D *fv_cell = m.getCell( c );

		//cell.velocity[0] = V[ c ].x;
		//cell.velocity[1] = V[ c ].y;
		//cell.polution = pol[ c ];
		polutions[c] = pol[c];
		//cell.area = fv_cell->area;
		areas[c] = fv_cell->area;
		//cell.init( fv_cell->nb_edge );
		indexes[c] = cell_edge_count;
		cell_edge_count += fv_cell->nb_edge;
		//for ( unsigned e = 0 ; e < cell.edge_count ; ++e )
		//	cell.edges[ e ] = fv_cell->edge[ e ]->label - 1;
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
	time=0.;nbiter=0;
	FVio pol_file( pol_fname.c_str() ,FVWRITE);
	//FVio pol_file("polution.omp.xml",FVWRITE);
	//pol_file.put(pol,time,"polution"); 
	//cout<<"computing"<<endl;
	while(time<final_time)
//	for ( int i = 0 ; i < 10 ; ++i )
	{
//		cout << "--[" << time << "]--" << endl;

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
	//    nbiter++;
	//    if(nbiter%nbjump==0)
	//        {
	//        pol_file.put(pol,time,"polution");    
	//        printf("step %d  at time %f \r",(int)nbiter,time); fflush(NULL);
	//        }
	// 
		{
//			using std::cout;
//			using std::endl;
//			for ( int j = 0 ; j < cell_count ; ++j )
//				cout
//					<<	'['	<<	j	<<	"]:"	<<	cells[j].polution	<<	endl;
//			getchar();
		}
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
	cout
#if   defined (PROFILE_MEMORY)
#if   defined (PROFILE_BTM)
				<<	btm_bytes
#endif//    PROFILE_BTM
#endif//    PROFILE_MEMORY
						<<	endl
		;
#endif//    PROFILE
}
