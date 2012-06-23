#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
#include "FVL/FVXMLReader.h"
#include "FVL/FVXMLWriter.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
using namespace std;

#ifdef NO_CUDA
	#include "kernels_cpu.h"
#else
	#define  _CUDA 1
	#include <cuda.h>
	#include "kernels_cuda.cuh"
#endif

#define BLOCK_SIZE_FLUX				512
#define BLOCK_SIZE_UPDATE			512
#define GRID_SIZE(elems, threads)	((int) std::ceil((double)elems/threads))

#define _CUDA_ONLY      if (_CUDA)
#define _NO_CUDA_ONLY   if (!_CUDA)

/**
 * Parameters struct passed via xml file
 */
struct Parameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	double anim_time;
	int anim_jump;
	double dirichlet;
	double CFL;

	public:
	// Constructor receives parameter file
	Parameters(string parameters_filename) {
		FVL::FVParameters para(parameters_filename);

		this->mesh_file		= para.getString("MeshName");
		this->velocity_file	= para.getString("VelocityFile");
		this->initial_file	= para.getString("PoluInitFile");
		this->output_file	= para.getString("OutputFile");
		this->final_time	= para.getDouble("FinalTime");
		this->anim_time		= para.getDouble("AnimTimeStep");
		this->anim_jump		= para.getInteger("NbJump");
		this->dirichlet		= para.getDouble("DirichletCondition");
		this->CFL			= para.getDouble("CFL");
	}
};

int main(int argc, char **argv) {
	#ifdef PROFILE
		PROFILE_INIT();
	#endif

	#ifdef PROFILE_ZONES
		PROFILE_START();
	#endif

	// print cuda mode
	#ifndef PROFILE
		#ifdef _CUDA
			cout << "CUDA mode: enabled" << endl;
		#else
			cout << "CUDA mode: disabled" << endl;
		#endif
	#endif

	// var declaration
	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	string param_filename;
	if (argc != 2) {
		param_filename = "param.xml";
	} else
		param_filename = argv[1];

	Parameters data(param_filename);

	// read mesh
	FVL::CFVMesh2D           mesh(data.mesh_file);
	FVL::CFVArray<double>    polution(mesh.num_cells);		// polution arrays
	FVL::CFVArray<double>    flux(mesh.num_edges);			// flux array
	FVL::CFVPoints2D<double> velocities(mesh.num_cells);	// velocities by cell (to calc vs array)
	FVL::CFVArray<double>    vs(mesh.num_edges);			// velocities by edge
	#ifdef OPTIM_LENGTH_AREA_RATIO
		FVL::CFVMat<double>      length_area_ratio(MAX_EDGES_PER_CELL, 1, mesh.num_cells);
	#endif

	// read other input files
	FVL::FVXMLReader velocity_reader(data.velocity_file);
	FVL::FVXMLReader polu_ini_reader(data.initial_file);
	velocity_reader.getPoints2D(velocities, t, name);
	polu_ini_reader.getVec(polution, t, name);
	polu_ini_reader.close();
	velocity_reader.close();

	FVL::FVXMLWriter polution_writer(data.output_file);
	polution_writer.append(polution, t, "polution");

	// compute velocity vector
	// TODO: Convert to CUDA
	#ifdef _CUDA
		kernel_compute_edge_velocities(mesh, velocities, vs, v_max);
		h = kernel_compute_mesh_parameter(mesh);

		#ifdef OPTIM_LENGTH_AREA_RATIO
			kernel_compute_length_area_ratio(mesh, length_area_ratio);
		#endif

	#else
		cpu_compute_edge_velocities(mesh, velocities, vs, v_max);
		h = cpu_compute_mesh_parameter(mesh);
		
		#ifdef OPTIM_LENGTH_AREA_RATIO
			cpu_compute_length_area_ratio(mesh, length_area_ratio);
		#endif
	#endif

	dt	= 1.0 / v_max * h;

	#ifdef _CUDA
		// saves whole mesh to CUDA memory
		mesh.cuda_malloc();
		polution.cuda_malloc();
		flux.cuda_malloc();
		vs.cuda_malloc();

		#ifdef OPTIM_LENGTH_AREA_RATIO
			length_area_ratio.cuda_malloc();
		#endif

		// data copy
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		mesh.cuda_save(stream);
		polution.cuda_save(stream);
		vs.cuda_save(stream);

		#ifdef OPTIM_LENGTH_AREA_RATIO
			length_area_ratio.cuda_save(stream);
		#endif
	
		// block and grid sizes for each kernel
		dim3 grid_flux(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_FLUX), 1, 1);
		dim3 block_flux(BLOCK_SIZE_FLUX, 1, 1);

		dim3 grid_update(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_UPDATE), 1, 1);
		dim3 block_update(BLOCK_SIZE_UPDATE, 1, 1);
	#endif

	// loop control vars
	bool   finished       = false;
	double anim_next_step = data.anim_time;

	// PROFILE ZONES --- measure preprocessing time
	#ifdef PROFILE_ZONES
		#ifdef _CUDA
			cudaDeviceSynchronize();
		#endif
		PROFILE_STOP();
		PROFILE_RETRIEVE_PRE();
		PROFILE_START();
	#endif

	//
	// main loop start
	//
	#ifdef PROFILE_ITERATION_CAP
	for(unsigned int iter = 0; iter < MAX_ITERATIONS; ++iter) {
	#else
	while (!finished) {
	#endif
		#ifndef PROFILE
			cout << "time: " << t << "   iteration: " << i << "\r";
		#endif

		if (t + dt > data.final_time) {
			#ifndef PROFILE
				cout << endl << "Final iteration, adjusting dt" << endl;
			#endif
			dt = data.final_time - t;
			finished = true;
		}

		#ifdef _CUDA
			#ifdef OPTIM_KERNELS
				kernel_compute_flux_optim<<< grid_flux, block_flux >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), flux.cuda_get(), data.dirichlet);
				kernel_update_optim<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt, length_area_ratio.cuda_get());

			#elif defined(OPTIM_LENGTH_AREA_RATIO)
				kernel_compute_flux<<< grid_flux, block_flux >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), flux.cuda_get(), data.dirichlet);
				kernel_update2<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt, length_area_ratio.cuda_get());

			#else
				kernel_compute_flux<<< grid_flux, block_flux >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), flux.cuda_get(), data.dirichlet);
				kernel_update<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt);
			#endif

			_DEBUG cudaCheckError(string("kernels"));

		#else
			cpu_compute_flux(mesh, vs, polution, flux, data.dirichlet); // compute_flux
			#ifdef OPTIM_LENGTH_AREA_RATIO
				cpu_update_optim(mesh, polution, flux, dt, length_area_ratio);
			#else
				cpu_update(mesh, polution, flux, dt);						// update
			#endif
		#endif
		
		t += dt;

		// anim save
		if (t >= anim_next_step) {
			#ifdef _CUDA
				polution.cuda_load();
			#endif
			
			polution_writer.append(polution, t, "polution");
			anim_next_step += data.anim_time;
		}

		++i;
	}

	// PROFILE ZONES --- measure main loop time
	#ifdef PROFILE_ZONES
		#ifdef _CUDA
			cudaDeviceSynchronize();
		#endif
		PROFILE_STOP();
		PROFILE_RETRIEVE_MAIN_LOOP();
		PROFILE_START();
	#endif

	// last anim save
	#ifdef _CUDA
		polution.cuda_load();
	#endif
	polution_writer.append(polution, t, "polution");

	// cleanup
	polution_writer.save();
	polution_writer.close();

	#ifdef _CUDA
		vs.cuda_free();
		polution.cuda_free();
		flux.cuda_free();

		_DEBUG cudaCheckError(string("final check"));
	#endif

	// PROFILE ZONE --- measure postprocessing time
	#ifdef PROFILE_ZONES
		PROFILE_STOP();
		PROFILE_RETRIEVE_POS();
		PROFILE_OUTPUT();
	#endif

	#ifdef PROFILE
		PROFILE_CLEANUP();
	#endif

}

