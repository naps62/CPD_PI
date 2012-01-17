#include <iostream>
#include <cfloat>
#include <omp.h>
#include "FVLib.h"
#include "MFVLog.h"

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

double *vs;

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
	double v=0;								//	resulting velocity
	double v_max;							//	maximum computed velocity
	FVEdge2D *edge;							//	current edge

	//for ( mesh.beginEdge(); ( edge = mesh.nextEdge() ) ; )
	es = mesh.getNbEdge();
	//#pragma omp parallel for
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
		//TODO: remove this dependence
		//if ( ( abs(v) * dt ) > 1)
		//	dt = 1.0 / abs(v);
		vs[e] = v;
		//end dependence
		if ( v < 0 )
			flux[ edge->label - 1 ] = v * p_right;
		else
			flux[ edge->label - 1 ] = v * p_left;
	}
	v_max = DBL_MIN;
	for ( e = 0; e < es; ++e)
	{
		cout << e << " vs= " << vs[e];
		getchar();
		if ( vs[e] > v_max )
			v_max = vs[e];
	}

	dt = 1.0 / abs(v_max);
		
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

	//for ( mesh.beginEdge(); ( edge = mesh.nextEdge() ) ; )
	int es = mesh.getNbEdge();
	//#pragma omp parallel for
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
	data.filenames.polution.output = "polution.openmp.xml";
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
	//int cs = mesh.getNbCell();
	//for (int c = 0; c < cs; ++c)
	{
		//cell = mesh.getCell(c);
		S = cell->area;
		for ( cell->beginEdge(); ( edge = cell->nextEdge() ) ; )
		//int es = cell->getNbEdge();
		//for (int e = 0; e < es; ++e)
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
	double dc)								//	Dirichlet condition
{
	double t;								//	time elapsed
	double dt;
	int i;									//	current iteration
	FVio polution_file("polution.xml",FVWRITE);

	t = 0;
	i = 0;
	polution_file.put( polutions , t , "polution" ); 


	cout
		<< "computing"
		<< endl;
	while ( t < final_time )
	{
		dt = compute_flux( mesh , polutions , velocities , fluxes , dc ) * mesh_parameter;
		update( mesh , polutions , fluxes , dt );
		t += dt;
		++i;
		if ( i % jump_interval == 0 )
		{
			polution_file.put( polutions , t , "polution" );    
			printf("step %d  at time %f \r", i, t);
			fflush(NULL);
		}
	}
	polution_file.put( polutions , t , "polution" ); 
}

/*
	Função Madre
*/
int main()
{  
	string name;
	double h;
	double t;
	FVMesh2D mesh;
	Parameters data;

	// read the parameter
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

	//	prepare velocities array
	vs = new double[ mesh.getNbEdge() ];

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
		data.computation.threshold)
	;
}

