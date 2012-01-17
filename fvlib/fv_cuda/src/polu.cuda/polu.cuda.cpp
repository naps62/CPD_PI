#include <iostream>
#include <cmath>
#include <cfloat>
#include "FVLib.h"
#include "CUDA/CFVLib.h"

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
		fv_float final;
	} time;
	struct
	{
		int jump;
	} iterations;
	struct
	{
		fv_float threshold;
	} computation;
}
Parameters;

//	END TYPES

//	BEGIN GLOBAL VARIABLES

fv_float *vs;
CudaFV::CFVVect<fv_float> gpu_vs;

//	END GLOBAL VARIABLES

//	BEGIN FUNCTIONS

/*
	Computes the resulting flux in every edge
*/
fv_float gpu_compute_flux(
		CudaFV::CFVMesh2D &mesh,
		CudaFV::CFVVect<fv_float> &polution,
		CudaFV::CFVPoints2D &velocity,
		CudaFV::CFVVect<fv_float> &flux,
		fv_float dc) {
	fv_float dt;
	fv_float p_left;							//	polution in the left face
	fv_float p_right;							//	polution in the right face
	int i_left;								//	index of the left face
	int i_right;							//	index of the right face
	unsigned e;								//	edge iteration variable
	fv_float v_left[2];						//	velocity in the left face
	fv_float v_right[2];						//	velocity in the right face
	fv_float v=0;								//	resulting velocity
	fv_float v_max;							//	maximum computed velocity

	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		i_left = mesh.edge_left_cells[i];
		v_left[0] = velocity.x[i_left];
		v_left[1] = velocity.y[i_left];
		p_left = polution[i_left];

		if (mesh.edge_right_cells[i] != NO_RIGHT_EDGE) {
			i_right = mesh.edge_right_cells[i];
			v_right[0] = velocity.x[i_right];
			v_right[1] = velocity.y[i_right];
			p_right = polution[i_right];
		} else {
			v_right[0] = v_left[0];
			v_right[1] = v_left[1];
			p_right = dc;
		}

		fv_float vx = (v_left[0] + v_right[0]) * 0.5 * mesh.edge_normals.x[i];
		fv_float vy = (v_left[1] + v_right[1]) * 0.5 * mesh.edge_normals.y[i];
		v = vx + vy;

		gpu_vs[i] = v;

		if (v < 0)
			flux[i] = v * p_right;
		else
			flux[i] = v * p_left;
	}

	v_max = -1;
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		cout << i << " vs= " << gpu_vs[i];
		getchar();
		if (gpu_vs[i] > v_max || v_max < 0)
			v_max = gpu_vs[i];
	}

	dt = 1.0 / abs(v_max);

	return dt;
}


void gpu_update(CudaFV::CFVMesh2D &mesh, CudaFV::CFVVect<fv_float> &polution, CudaFV::CFVVect<fv_float> &flux, fv_float dt) {
	for (unsigned int i = 0; i < mesh.num_edges; ++i) {
		polution[ (unsigned int) mesh.edge_left_cells[i] ] -=
			dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_left_cells[i] ];
		if (mesh.edge_right_cells[i] != NO_RIGHT_EDGE)
			polution[ (unsigned int) mesh.edge_right_cells[i] ] +=
				dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_right_cells[i] ];
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
fv_float compute_mesh_parameter (
	FVMesh2D mesh)
{
	fv_float h;
	fv_float S;
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

//fv_float gpu_compute_mesh_parameter (GPU_FVMesh2D mesh) {
	//fv_float h, S;
	//for(unsigned int i = 0; i < mesh.num_cells; ++i) {
	//	S = mesh.cell_areas.cpu_ptr[i];
		//TODO continuar aqui. cada Cell precisa da lista de edges correspondentes
	//}
//	return 0;
//}

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
void gpu_main_loop(fv_float final_time, unsigned jump_interval, CudaFV::CFVMesh2D &mesh, fv_float mesh_parameter, FVVect<fv_float> old_polution, CudaFV::CFVVect<fv_float> &polutions, CudaFV::CFVPoints2D &velocities, CudaFV::CFVVect<fv_float> &flux, fv_float dc) {
	fv_float t, dt;
	int i;
	FVio polution_file("polution.xml", FVWRITE);

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
	        for(unsigned int x = 0; x< mesh.num_cells; ++x) {
        		old_polution[x] = polutions[x];
        	}
			polution_file.put(old_polution, t, "polution");
			printf("step %d at time %f \r", i, t);
			fflush(NULL);
		}
	}

	for(unsigned int x = 0; x < mesh.num_cells; ++x) {
		old_polution[x] = polutions[x];
	}
	polution_file.put(old_polution, t, "polution");
}

void cuda_main_loop(
		fv_float final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		fv_float mesh_parameter,
		FVVect<fv_float> &old_polution,
		CudaFV::CFVVect<fv_float> &polution,
		CudaFV::CFVPoints2D &velocities,
		CudaFV::CFVVect<fv_float> &flux,
		fv_float dc);
/*
	Função Madre
*/
int main()
{  
	string name;
	fv_float h;
	fv_float t;
	FVMesh2D mesh;

	// GPU

	Parameters data;

	// read the parameter
	data = read_parameters( "param.xml" );

	// read the mesh
	mesh.read( data.filenames.mesh.c_str() );

	// GPU
    CudaFV::CFVMesh2D gpu_mesh(mesh);
	
	FVVect<fv_float> polution( mesh.getNbCell() );
	FVVect<fv_float> flux( mesh.getNbEdge() );
	FVVect<FVPoint2D<fv_float> > velocity( mesh.getNbCell() );

	//	read velocity
	FVio velocity_file( data.filenames.velocity.c_str() , FVREAD );
	velocity_file.get( velocity , t , name );

	// GPU
    CudaFV::CFVVect<fv_float> gpu_polution(gpu_mesh.num_cells);
    CudaFV::CFVVect<fv_float> gpu_flux(gpu_mesh.num_edges);
    CudaFV::CFVPoints2D gpu_velocity(gpu_mesh.num_cells);

	for(unsigned int i = 0; i < gpu_polution.size(); ++i) {
		gpu_polution[i] = polution[i];
		gpu_velocity.x[i] = velocity[i].x;
		gpu_velocity.y[i] = velocity[i].y;
	}

	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVREAD );
	polu_ini_file.get( polution , t , name );

	gpu_vs = CudaFV::CFVVect<fv_float>(gpu_mesh.num_edges);

	// TODO implementar a versao GPU disto
	h = compute_mesh_parameter( mesh );
	//h = gpu_compute_mesh_parameter(gpu_mesh);

	// GPU
	cuda_main_loop(
		data.time.final,
		data.iterations.jump,
		gpu_mesh,
		h,
		polution,
		gpu_polution,
		gpu_velocity,
		gpu_flux,
		data.computation.threshold);

	return 0;
}
