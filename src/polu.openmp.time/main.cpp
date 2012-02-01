#include <iostream>
#include <limits>

#include "FVLib.h"

#include <fv/cpu/cell.hpp>
#include <fv/cpu/edge.hpp>

using fv::cpu::Cell;
using fv::cpu::Edge;

//	BEGIN CONSTANTS

#define	OMP_FRC_ALL	1

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

double *max_vel_v;
int tc;

//	END GLOBAL VARIABLES

//	BEGIN FUNCTIONS

/*
	Computes the resulting flux in every edge
*/
double compute_flux(
	Edge *edges,							//	edges vector
	unsigned edgec,							//	number of edges in the vector
	Cell *cells,							//	cells vector
	double dirichlet)						//	Dirichlet condition
{
	double dt;
//	double p_left;							//	polution in the left face
//	double p_right;							//	polution in the right face
//	int i_left;								//	index of the left face
//	int i_right;							//	index of the right face
	int t;									//	current thread number
//	unsigned e;								//	edge iteration variable
//	unsigned es;							//	total number of edges
//	FVPoint2D<double> v_left;				//	velocity in the left face
//	FVPoint2D<double> v_right;				//	velocity in the right face
//	double v;								//	resulting velocity
//	FVEdge2D *edge;							//	current edge

	#pragma omp parallel	\
		default(shared)	\
		private(t)
	{
		unsigned e;
		unsigned i_r;
		double max_vel_t = numeric_limits<double>::min();
		double p_l;
		double p_r;
		double v;
		double v_l[2];
		double v_r[2];
//		Cell &cell_l;
//		Cell &cell_r;
//		Edge &edge;

		t = omp_get_thread_num();

		#pragma omp for
		for (e = 0; e < edgec; ++e)
		{
			Edge &edge = edges[e];
			
			//	left data
			Cell &cell_l = cells[edge.left];
			v_l[0] = cell_l.velocity[0];
			v_l[1] = cell_l.velocity[1];
			p_l = cell_l.polution;

			//	right data
			i_r = edge.right;
			if ( i_r == numeric_limits<unsigned>::max() )
			{
				v_r[0] = v_l[0];
				v_r[1] = v_l[1];
				p_r = dirichlet;
			}
			else
			{
				Cell &cell_r = cells[i_r];
				v_r[0] = cell_r.velocity[0];
				v_r[1] = cell_r.velocity[1];
				p_r = cell_r.polution;
			}

			v = (v_l[0] + v_r[0]) * 0.5 * edge.normal[0]
			  + (v_l[1] + v_r[1]) * 0.5 * edge.normal[1];

			edge.flux =	( v < 0 )
						? ( v * p_r )
						: ( v * p_l );

			max_vel_t =	( v > max_vel_t )
						? v
						: max_vel_t;
		}
	}

	double max_vel = numeric_limits<double>::min();
	for (t = 0; t < tc; ++t)
		max_vel =	( max_vel_v[t] > max_vel )
					? max_vel_v[t]
					: max_vel;
	
	dt = 1.0 / abs( max_vel );

	return dt;
}

/*
	Updates the polution values based on the flux through every edge.
*/
void update(
	Cell *cells,
	unsigned cellc,
	Edge *edges,
	double dt)
{
	unsigned c;
	unsigned e;
//	Cell &cell;
//	Edge &edge;

	for ( c = 0 ; c < cellc ; ++c )
	{
		Cell &cell = cells[c];
		for ( e = 0 ; e < cell.edgec ; ++e )
		{
			Edge &edge = edges[cell.edges[e]];
			if ( c == edge.left )
				cell.polution -= dt * edge.flux * edge.length / cell.area;
			else
				cell.polution += dt * edge.flux * edge.length / cell.area;
		}
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
//	FVMesh2D mesh,							//	2D mesh to compute
	double mesh_parameter,					//	mesh parameter
//	FVVect<double> polutions,				//	polution values vector
//	FVVect<FVPoint2D<double> > velocities,	//	velocity vectors collection
//	FVVect<double> fluxes,					//	flux values vector

	Cell *cells,
	unsigned cellc,
	Edge *edges,
	unsigned edgec,
	
	double dirichlet,						//	Dirichlet condition
	string output_filename)					//	output file name
{
	double t = 0;							//	time elapsed
	double dt;								//	instant duration






//	FVio polution_file( output_filename.c_str() ,FVWRITE);
											//	output file




	



	//for ( t = 0 ; t < final_time ; t += dt )
	//{
		dt = compute_flux(edges,edgec,cells,dirichlet) * mesh_parameter;
	//	update(cells,cellc,edges,dt);
	//}


	{
		using std::cout;
		using std::endl;
		cout
			<<	"dt: "	<<	dt	<<	endl;
	}


//	FVVect<double> polution( cellc );
//	for ( unsigned c = 0 ; c < cellc ; ++c )
//		polution[c] =  cells[c].polution;
//
//
//
//	polution_file.put( polution , t , "polution" ); 



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



	//	OpenMP init
	tc = omp_get_num_procs() * OMP_FRC_ALL;

	//	Data init
	max_vel_v = new double[tc];
	//		cells
	unsigned cellc = mesh.getNbCell();
	Cell *cells = new Cell[cellc];
	for (unsigned c = 0; c < cellc; ++c)
	{
		Cell &cell = cells[c];
		cell.velocity[0] = velocity[c].x;
		cell.velocity[1] = velocity[c].y;
		cell.polution = polution[c];
		FVCell2D *fv_cell = mesh.getCell(c);
		cell.area = fv_cell->area;
		cell.init( fv_cell->nb_edge );
		for (unsigned e = 0 ; e < cell.edgec ; ++e )
			cell.edges[e] = fv_cell->edge[e]->label - 1;
	}

	unsigned edgec = mesh.getNbEdge();
	Edge *edges = new Edge[ edgec ];
	for ( unsigned e = 0 ; e < edgec ; ++e )
	{
		Edge &edge = edges[ e ];
		edge.flux = flux[ e ];
		FVEdge2D *fv_edge = mesh.getEdge( e );
		edge.left = fv_edge->leftCell->label - 1;
		if (fv_edge->rightCell)
			edge.right = fv_edge->rightCell->label - 1;
		else
			edge.right = numeric_limits<unsigned>::max();
		edge.length = fv_edge->length;
	}
		


	

	// compute the Mesh parameter
	h = compute_mesh_parameter( mesh );

	// the main loop
//	main_loop(
//		data.time.final,
//		data.iterations.jump,
//		mesh,
//		h,
//		polution,
//		velocity,
//		flux,
//		data.computation.threshold,
//		data.filenames.polution.output)
//	;
	main_loop(
		data.time.final,
		h,
		cells,
		cellc,
		edges,
		edgec,
		data.computation.threshold,
		data.filenames.polution.output)
	;

	delete[] max_vel_v;
	delete[] cells;
	delete[] edges;

	return 0;
}
