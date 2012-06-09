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
	#else
		cpu_compute_edge_velocities(mesh, velocities, vs, v_max);
		h = cpu_compute_mesh_parameter(mesh);
	#endif

	dt	= 1.0 / v_max * h;

	#ifdef _CUDA
		// saves whole mesh to CUDA memory
		mesh.cuda_malloc();
		polution.cuda_malloc();
		flux.cuda_malloc();
		vs.cuda_malloc();

		// data copy
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		mesh.cuda_save(stream);
		polution.cuda_save(stream);
		vs.cuda_save(stream);
	
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
	while (!finished) {
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
			kernel_compute_flux<<< grid_flux, block_flux >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), flux.cuda_get(), data.dirichlet);
			_DEBUG cudaCheckError(string("compute_flux"));
	
			kernel_update<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), data.dirichlet);
			_DEBUG cudaCheckError(string("update"));
		#else
			cpu_compute_flux(mesh, vs, polution, flux, data.dirichlet); // compute_flux
			cpu_update(mesh, polution, flux, dt);						// update
		#endif
		
		t += dt;

		// anim save
		if (t >= anim_next_step || 1==1) {
			#ifdef _CUDA
				polution.cuda_load();
				flux.cuda_load();
				//mesh.edge_lengths.cuda_load();
				//mesh.cell_areas.cuda_load();
			#endif
			for(unsigned int i = 0; i < 50; ++i)
				cout << i << " " << polution[i] << endl;
				//cout << "edge_lengths[" << i << "] = " << mesh.edge_lengths[i] << " " << "cell_area[" << i << " ] = " << mesh.cell_areas[i] << endl;

			//polution_writer.append(polution, t, "polution");
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

