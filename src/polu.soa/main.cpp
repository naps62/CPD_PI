#include <iostream>
#include <limits>
#include "FVLib.h"





void
compute_flux
	(
	double *   polutions,
	double *   velocities,
	unsigned * cell_left,
	unsigned * cell_right,
	double *   fluxes,
	double     dirichlet,
	unsigned   edge_count
	)
{
	double     polution_left;
	double     polution_right;

	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		double polution_left = polutions[ cell_left[e] ];
		double polution_right
			= ( cell_right[e] < numeric_limits<unsigned>::max() )
			? cell_right[e]
			: dirichlet
			;
		fluxes[e] = ( velocities[e] < 0 )
		          ? velocities[e] * polution_right
				  : velocities[e] * polution_left
				  ;
	}
}






void
update
	(
	double *   areas,
	double *   fluxes,
	double *   lengths,
	unsigned * indexes,
	unsigned * edges,
	unsigned * cell_left,
	double     dt,
	unsigned   index_count,
	unsigned   cell_count
	)
{
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
			if ( cell_left[e] == c )
				cdp -= edp;
			else
				cdp += edp;
		}
	}
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
	//Edge *edges = new Edge[ edge_count ];
	double *   fluxes     = new double[ edge_count ];
	double *   lengths    = new double[ edge_count ];
	double *   velocities = new double[ edge_count ];
	unsigned * cell_left  = new unsigned[ edge_count ];
	unsigned * cell_right = new unsigned[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		//Edge &edge = edges[ e ];
		FVEdge2D *fv_edge = m.getEdge( e );

		//edge.flux = flux[ e ];
		fluxes[ e ]   = flux[ e ];
		//edge.length = fv_edge->length;
		lengths[ e ]  = fv_edge->length;
		//edge.left = fv_edge->leftCell->label - 1;
		cell_left[e]  = fv_edge->leftCell->label - 1;
		//edge.right = ( fv_edge->rightCell )
		cell_right[e] = ( fv_edge->rightCell )
					  ? fv_edge->rightCell->label - 1
					  : numeric_limits<unsigned>::max();

		double normal[2];
		normal[0] = fv_edge->normal.x;
		normal[1] = fv_edge->normal.y;
		double v_left[2];
		v_left[0] = V[ cell_left[e] ].x;
		v_left[1] = V[ cell_left[e] ].y;
		double v_right[2];
		//if ( edge.right < numeric_limits<unsigned>::max() )
		if ( cell_right[e] < numeric_limits<unsigned>::max() )
		{
			v_right[0] = V[ cell_right[e] ].x;
			v_right[1] = V[ cell_right[e] ].y;
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
	double *   polution        = new double[ cell_count ];
	double *   areas           = new double[ cell_count ];
	unsigned * indexes         = new unsigned[ cell_count ];
	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		//Cell &cell = cells[ c ];
		FVCell2D *fv_cell = m.getCell( c );

		//cell.velocity[0] = V[ c ].x;
		//cell.velocity[1] = V[ c ].y;
		//cell.polution = pol[ c ];
		polution[c] = pol[c];
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
	}
			








	// the main loop
	time=0.;nbiter=0;
	FVio pol_file("polution.omp.xml",FVWRITE);
	//pol_file.put(pol,time,"polution"); 
	//cout<<"computing"<<endl;
	while(time<final_time)
//	for ( int i = 0 ; i < 10 ; ++i )
	{
		compute_flux(
			polution,
			velocities,
			cell_left,
			cell_right,
			fluxes,
			dirichlet,
			edge_count)
		;
		update(
			areas,
			fluxes,
			lengths,
			indexes,
			edges,
			cell_left,
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
		pol[ c ] = polution[c];

	pol_file.put(pol,time,"polution"); 

	delete[] fluxes;
	delete[] lengths;
	delete[] velocities;
	delete[] cell_left;
	delete[] cell_right;
	delete[] polution;
	delete[] areas;
	delete[] indexes;
	delete[] edges;
}
