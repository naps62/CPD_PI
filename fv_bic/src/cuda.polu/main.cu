#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
#include "FVL/FVXMLReader.h"
#include "FVL/FVXMLWriter.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
using namespace std;

#ifdef NO_CUDA
	#define  _CUDA 0
	#include "kernels_cpu.h"
#else
	#define  _CUDA 1
	#include <cuda.h>
	#include "kernels_cuda.cuh"

	#define BLOCK_SIZE_FLUX				512
	#define BLOCK_SIZE_UPDATE			512
	#define GRID_SIZE(elems, threads)	((int) std::ceil((double)elems/threads))
#endif

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
	void Parameters(string parameters_filename) {
		FVL::FVParameters para(parameters_filename);

		this->mesh_file		= para->getString("MeshName");
		this->velocity_file	= para->getString("VelocityFile");
		this->initial_file	= para->getString("PoluInitFile");
		this->output_file	= para->getString("OutputFile");
		this->final_time	= para->getDouble("FinalTime");
		this->anim_time		= para->getDouble("AnimTimeStep");
		this->anim_jump		= para->getInteger("NbJump");
		this->dirichlet		= para->getDouble("DirichletCondition");
		this->CFL			= para->getDouble("CFL");
	}
};

int main(int argc, char **argv) {

	// print cuda mode
	if (_CUDA)
		cout << "CUDA mode: disabled" << endl;
	else
		cout << "CUDA mode: enabled" << endl;

	// var declaration
	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	Parameters data;
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
	cpu_compute_edge_velocities(mesh, velocities, vs, v_max);

	h = cpu_compute_mesh_parameter(mesh);
	dt	= 1.0 / v_max * h;

	if (_CUDA) {
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
	}

	// loop control vars
	bool   finished       = false;
	double anim_next_step = data.anim_time;

	//
	// main loop start
	//
	while (!finished) {
		cout << "time: " << t << "   iteration: " << i << "\r";

		if (t + dt > data.final_time) {
			cout << endl << "Final iteration, adjusting dt" << endl;
			dt = data.final_time - t;
			finished = true;
		}

		if (_CUDA) {
			kernel_compute_flux<<< grid_flux, block_flux >>>(mesh.cuda_get(), polution.cuda_get(), vs.cuda_get(), flux.cuda_get(), data.dirichlet);
			_DEBUG cudaCheckError(string("compute_flux"));
	
			kernel_update<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), data.dirichlet);
			_DEBUG cudaCheckError(string("update"));
		} else {
			cpu_compute_flux(mesh, vs, polution, flux, data.dirichlet); // compute_flux
			cpu_update(mesh, polution, flux, dt);						// update
		}
		
		t += dt;

		// anim save
		if (t >= anim_next_step) {
			if (_CUDA) polution.cuda_get();
			polution_writer.append(polution, t, "polution");
			anim_next_step += data.anim_time;
		}

		++i;
	}

	// last anim save
	if (_CUDA) polution.cuda_get();
	polution_writer.append(polution, t, "polution");

	// cleanup
	polution_writer.save();
	polution_writer.close();

	if (_CUDA) {
		vs.cuda_free();
		polution.cuda_free();
		flux.cuda_free();
		mesh.cuda_free();

	_DEBUG cudaCheckckErroeError(string("final check"));
	}
}

