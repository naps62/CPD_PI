#include "FVL/FVLib.h"
using namespace std;

#ifdef NO_CUDA
#include "kernels_cpu.h"
#else
#include <cuda.h>
#include "kernels_cuda.cuh"
#endif

#define BLOCK_SIZE_FLUX		512
#define BLOCK_SIZE_UPDATE	512
#define GRID_SIZE(elems, threads)	((int) std::ceil((double)elems/threads))

typedef struct _parameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	double anim_time;
	int anim_jump;
	double dirichlet;
	double CFL;
} Parameters;


#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>
#include <set>

void prepare_mesh_test_data(CFVMesh2D &mesh, CFVArray<double> &polution) {
	double min_x = std::numeric_limits<double>::max();
	double max_x = std::numeric_limits<double>::min();

	/* find min and max x coords of the mesh edges */
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		double current = mesh.edge_centroids.x[i];
		if (current < min_x) min_x = current;
		if (current > max_x) max_x = current;
	}

	cout << endl << "Linking mesh ends" << endl;
	/* This assumes the mesh is rectangular, and we want to connect the left side with the right side
	 * that is, for every edge E with x = min_x, and no right cell, make the right cell equal to the left cell of the corresponding edge on the right side, and vice-versa
	 **/
	set<unsigned int> left_cells;
	set<unsigned int> right_cells;

	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		if (mesh.edge_types[i] == FV_EDGE_DIRICHLET) {
			if (mesh.edge_centroids.x[i] == min_x)
				left_cells.insert(i);
			
			if (mesh.edge_centroids.x[i] == max_x)
				right_cells.insert(i);
		}
	}

	set<unsigned int>::iterator left_it, right_it;
	for(left_it = left_cells.begin(), right_it = right_cells.begin();
		left_it != left_cells.end();
		++left_it, ++right_it) {

		unsigned int l = *left_it;
		unsigned int r = *right_it;

		/* set edges type to regular */
		mesh.edge_types[l] = FV_EDGE_FAKE;
		mesh.edge_types[r] = FV_EDGE_FAKE;

		/* link both edges */
		mesh.edge_right_cells[l] = mesh.edge_left_cells[l];
		mesh.edge_left_cells[l]  = mesh.edge_left_cells[r];

		mesh.edge_right_cells[r] = mesh.edge_right_cells[l];
		cout << "linking edge " << l << " with " << r << endl;
	}

	cout << "Linked " << left_cells.size() << " pairs of edges " << endl << endl;
}

Parameters read_parameters (string parameters_filename) {
	Parameters data;
	FVParameters para(parameters_filename);

	data.mesh_file		= para.getString("MeshName");
	data.velocity_file	= para.getString("VelocityFile");
	data.initial_file	= para.getString("PoluInitFile");
	data.output_file	= para.getString("OutputFile");
	data.final_time		= para.getDouble("FinalTime");
	data.anim_time		= para.getDouble("AnimTimeStep");
	data.anim_jump		= para.getInteger("NbJump");
	data.dirichlet		= para.getDouble("DirichletCondition");
	data.CFL			= para.getDouble("CFL");

	return data;
}


int main(int argc, char **argv) {

	#ifdef NO_CUDA
		cout << "Running in NO_CUDA mode" << endl;
	#endif

	// var declaration
	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	Parameters data;
	if (argc != 2) {
		cerr << "Arg warning: no xml param filename specified. Defaulting to param.xml" << endl;
		data = read_parameters("param.xml");
	} else
		data = read_parameters(argv[1]);

	// read mesh
	FVL::CFVMesh2D		mesh(data.mesh_file);
	FVL::CFVRecons2D	recons(mesh);

	FVL::CFVPoints2D<double> velocities(mesh.num_cells);
	FVL::CFVArray<double>    polution(mesh.num_cells);
	//FVL::CFVArray<double>    flux(mesh.num_edges);
	//FVL::CFVArray<double>    oldflux(mesh.num_edges);
	//FVL::CFVArray<bool>      invalidate_flux(mesh.num_cells);
	FVL::CFVArray<double>    vs(mesh.num_edges);
	FVL::CFVMat<double>      matA(2, 2, mesh.num_cells);
	FVL::CFVMat<double>      vecGradient(2, 1, mesh.num_cells);
	FVL::CFVMat<double>      vecR(2, 1, mesh.num_cells);
	FVL::CFVArray<double> edgePsi(mesh.num_edges);
	FVL::CFVArray<double> cellPsi(mesh.num_cells);

	// read other input files
	FVL::FVXMLReader velocity_reader(data.velocity_file);
	FVL::FVXMLReader polu_ini_reader(data.initial_file);
	polu_ini_reader.getVec(polution, t, name);
	velocity_reader.getPoints2D(velocities, t, name);
	polu_ini_reader.close();
	velocity_reader.close();

	/* assign test value for polution */
	prepare_mesh_test_data(mesh, polution);

	FVL::FVXMLWriter polution_writer(data.output_file);
	polution_writer.append(polution, t, "polution");

	// compute velocity vector
	// TODO: Convert to CUDA
	cpu_compute_edge_velocities(mesh, velocities, vs, v_max);
	h = cpu_compute_mesh_parameter(mesh);
	// TODO trocar 1.0 por parametro CFL (com valores entre 0 e 1, 0.3 para esquema de ordem 2)
	dt	= data.CFL / v_max * h;

	#ifndef NO_CUDA
	// saves whole mesh to CUDA memory
	mesh.cuda_malloc();
	recons.cuda_malloc();
	polution.cuda_malloc();
	candidate_polution.cuda_malloc();
	//flux.cuda_malloc();
	//oldflux.cuda_malloc();
	//invalidate_flux.cuda_malloc();
	vs.cuda_malloc();
	matA.cuda_malloc();
	vecGradient.cuda_malloc();
	vecR.cuda_malloc();
	edgePsi.cuda_malloc();
	cellPsi.cuda_malloc();

	// data copy
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	mesh.cuda_save(stream);
	polution.cuda_save(stream);
	vs.cuda_save(stream);
	

	// sizes of each kernel
	// TODO: mudar BLOCK_SIZE_FLUX para MAT_A
	dim3 grid_matA(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_matA(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_vecResult(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_vecResult(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_vecABC(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_vecABC(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_vecValidABC(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_vecValidABC(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_flux(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_flux(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_update(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_UPDATE), 1, 1);
	dim3 block_update(BLOCK_SIZE_UPDATE, 1, 1);
	#endif

	#ifdef NO_CUDA
		cpu_reverseA(mesh, matA);
	#else
		//kernel_compute_reverseA<<< grid_matA, block_matA >>>(mesh.cuda_get(), matA.cuda_get());
		//_D(cudaCheckError("cuda[compute_reverseA]"));
	#endif

	bool finished = false;
	double anim_next_step = data.anim_time;
	cout << "dt= " << dt << endl;
	//while(t <= data.final_time) {
	while (!finished) {
		cout << "time: " << t << "   iteration: " << i << '\r';
		
		if (t + dt > data.final_time) {
			cout << endl << "Final iteration, adjusting dt" << endl;
			dt = data.final_time - t;
			finished = true;
		}


		// Cpu version
		#ifdef NO_CUDA
			
			cpu_compute_vecR(mesh, polution, vecR, data.dirichlet);						// compute system polution coeficients for system solve
			cpu_compute_gradient(mesh, matA, vecR, vecGradient);						// compute (a,b,c) vector
			cpu_compute_u(mesh, recons, polution, vecGradient, t, dt);
			cpu_compute_border_u(mesh, recons, data.dirichlet);

			cpu_compute_unbounded_flux(mesh, recons, vs, polution, edgePsi, data.dirichlet);	// compute flux
			cpu_cellPsi(mesh, edgePsi, cellPsi);													// compute Psi bounder for each cell
			cpu_bound_u(mesh, recons, polution, vecGradient, cellPsi, t, dt);
			cpu_bound_flux(mesh, recons, vs, polution, data.dirichlet); 						// bound previously calculated flux using psi values
			cpu_update(mesh, recons, polution, dt); 

		#else

			/*kernel_compute_vecResult<<< grid_vecResult, block_vecResult >>>(mesh.cuda_get(), polution.cuda_get(), vecResult.cuda_get(), data.dirichlet);
			kernel_compute_vecABC<<< grid_vecABC, block_vecABC >>>(mesh, matA.cuda_get(), vecResult.cuda_get(), vecABC.cuda_get());
			kernel_compute_flux<<< grid_flux, block_flux >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), vecABC.cuda_get(), flux.cuda_get(), data.dirichlet, t, dt);
			kernel_update<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt);
			kernel_reset_oldflux<<< 512, 512 >>>(oldflux.cuda_get());
			kernel_detect_polution_errors<<< 512, 512 >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), oldflux.cuda_get(), invalidate_flux.cuda_get())
			kernel_fix_polution_errors<<< 512, 512 >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), flux.cuda_get(), oldflux.cuda_get(), invalidate_flux.cuda_get());
			kernel_fix_update<<< 512, 512 >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), oldflux.cuda_get(), dt, invalidate_flux.cuda_get());*/

		#endif

		t += dt;

		if (t >= anim_next_step) {
			#ifndef NO_CUDA
				polution.cuda_get();
			#endif

			polution_writer.append(polution, t, "polution");
			anim_next_step += data.anim_time;
		}
		++i;
	}

	polution_writer.save();
	polution_writer.close();

	#ifndef NO_CUDA
	polution.cuda_free();
	vs.cuda_free();
	vecGradient.cuda_free();
	vecR.cuda_free();
	matA.cuda_free();
	recons.cuda_free();
	mesh.cuda_free();
	#endif

	cout << endl << "exiting" << endl;
}

