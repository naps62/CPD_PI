#include <iostream>
#include <cfloat>
#include "FVLib.h"

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
FV_DualPtr gpu_vs;

//	END GLOBAL VARIABLES

//	BEGIN FUNCTIONS

/*
	Computes the resulting flux in every edge
*/
double gpu_compute_flux(GPU_FVMesh2D &mesh, FV_DualPtr &polution, FV_GPU_Point2D &velocity, FV_DualPtr flux, double dc) {
	double dt;
	double p_left;							//	polution in the left face
	double p_right;							//	polution in the right face
	int i_left;								//	index of the left face
	int i_right;							//	index of the right face
	unsigned e;								//	edge iteration variable
	double v_left[2];						//	velocity in the left face
	double v_right[2];						//	velocity in the right face
	double v=0;								//	resulting velocity
	double v_max;							//	maximum computed velocity

	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		i_left = mesh.edge_left_cells.cpu_ptr[i];
		v_left[0] = velocity.x.cpu_ptr[i_left];
		v_left[1] = velocity.y.cpu_ptr[i_left];
		p_left = polution.cpu_ptr[i_left];

		if (mesh.edge_right_cells.cpu_ptr[i] != NO_RIGHT_EDGE) {
			i_right = mesh.edge_right_cells.cpu_ptr[i];
			v_right[0] = velocity.x.cpu_ptr[i_right];
			v_right[1] = velocity.y.cpu_ptr[i_right];
		} else {
			v_right[0] = v_left[0];
			v_right[1] = v_left[1];
			p_right = dc;
		}

		double vx = v_left[0] + v_right[0] * 0.5 + mesh.edge_normals.x.cpu_ptr[i];
		double vy = v_left[1] + v_right[0] * 0.5 + mesh.edge_normals.y.cpu_ptr[i];
		v = vx + vy;

		gpu_vs.cpu_ptr[i] = v;

		if (v < 0)
			flux.cpu_ptr[i] = v * p_right;
		else
			flux.cpu_ptr[i] = v * p_left;
	}

	v_max = std::numeric_limits<double>::min();
	for(unsigned int i = 0; i < mesh.num_edges; ++e) {
		if (gpu_vs.cpu_ptr[i] > v_max)
			v_max = gpu_vs.cpu_ptr[i];
	}

	dt = 1.0 / abs(v_max);

	return dt;
}


void gpu_update(GPU_FVMesh2D &mesh, FV_DualPtr &polution, FV_DualPtr &flux, double dt) {
	for (unsigned int i = 0; i < mesh.num_edges; ++i) {
		
		polution.cpu_ptr[ (unsigned int) mesh.edge_left_cells.cpu_ptr[i] ] -=
			dt * flux.cpu_ptr[i] * mesh.edge_lengths.cpu_ptr[i] / mesh.cell_areas.cpu_ptr[ (unsigned int) mesh.edge_left_cells.cpu_ptr[i] ];
		if (mesh.edge_right_cells.cpu_ptr[i] != NO_RIGHT_EDGE)
			polution.cpu_ptr[ (unsigned int) mesh.edge_right_cells.cpu_ptr[i] ] +=
				dt * flux.cpu_ptr[i] * mesh.edge_lengths.cpu_ptr[i] / mesh.cell_areas.cpu_ptr[ (unsigned int) mesh.edge_right_cells.cpu_ptr[i] ];
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
	{
		S = cell->area;
		for ( cell->beginEdge(); ( edge = cell->nextEdge() ) ; )
		{
			if ( h * edge->length > S )
				h = S / edge->length;
		}
	}
	return h;
}

//double gpu_compute_mesh_parameter (GPU_FVMesh2D mesh) {
	//double h, S;
	//for(unsigned int i = 0; i < mesh.num_cells; ++i) {
	//	S = mesh.cell_areas.cpu_ptr[i];
		//TODO continuar aqui. cada Cell precisa da lista de edges correspondentes
	//}
//	return 0;
//}

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
void gpu_main_loop(double final_time, unsigned jump_interval, GPU_FVMesh2D &mesh, double mesh_parameter, FVVect<double> old_polution, FV_DualPtr &polutions, FV_GPU_Point2D &velocities, FV_DualPtr &flux, double dc) {
	double t, dt;
	int i;
	FVio polution_file("gpu_polution.xml", FVWRITE);

	t = 0;
	i = 0;
	polution_file.put(old_polution, t, "polution");
	cout << "computing" << endl;
	while(t < final_time) {
		dt = gpu_compute_flux(mesh, polutions, velocities, flux, dc) * mesh_parameter;
		gpu_update(mesh, polutions, flux, dt);
		t += dt;
		++i;
		if (i % jump_interval == 0) {
			//polution_file.put(polutions, t, "polution");
			printf("step %d at time %f \r", i, t);
			fflush(NULL);
		}
	}

	FVVect<double> polution(mesh.num_cells);
	for(unsigned int i = 0; i < mesh.num_cells; ++i) {
		polution[i] = polutions.cpu_ptr[i];
	}
	polution_file.put(polution, t, "polution");
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

	// GPU

	Parameters data;

	// read the parameter
	data = read_parameters( "param.xml" );

	// read the mesh
	mesh.read( data.filenames.mesh.c_str() );

	// GPU
	GPU_FVMesh2D gpu_mesh(mesh);

	FVVect<double> polution( mesh.getNbCell() );
	FVVect<double> flux( mesh.getNbEdge() );
	FVVect<FVPoint2D<double> > velocity( mesh.getNbCell() );

	//	read velocity
	FVio velocity_file( data.filenames.velocity.c_str() , FVREAD );
	velocity_file.get( velocity , t , name );

	// GPU
	FV_DualPtr gpu_polution(gpu_mesh.num_cells);
	FV_DualPtr gpu_flux(gpu_mesh.num_edges);
	FV_GPU_Point2D gpu_velocity(gpu_mesh.num_cells);

	for(unsigned int i = 0; i < gpu_polution.size; ++i) {
		gpu_polution.cpu_ptr[i] = polution[i];
		gpu_velocity.x.cpu_ptr[i] = velocity[i].x;
		gpu_velocity.y.cpu_ptr[i] = velocity[i].y;
	}

	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVREAD );
	polu_ini_file.get( polution , t , name );

	gpu_vs = FV_DualPtr(gpu_mesh.num_edges);

	// TODO implementar a versao GPU disto
	h = compute_mesh_parameter( mesh );
	//h = gpu_compute_mesh_parameter(gpu_mesh);

	// GPU
	gpu_main_loop(
		data.time.final,
		data.iterations.jump,
		gpu_mesh,
		h,
		polution,
		gpu_polution,
		gpu_velocity,
		gpu_flux,
		data.computation.threshold);
	unsigned es;							//	total number of edges
}
