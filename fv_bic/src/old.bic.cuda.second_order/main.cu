#include "FVL/FVLib.h"
#include "FVL/FVXMLWriter.h"
#include "FVL/FVArray.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
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
	int anim_jump;
	double dirichlet;
	double CFL;
} Parameters;

// TODO: interface decente para paremetros xml
Parameters read_parameters (string parameters_filename) {
	Parameters data;
	FVParameters para(parameters_filename);

	data.mesh_file		= para.getString("MeshName");
	data.velocity_file	= para.getString("VelocityFile");
	data.initial_file	= para.getString("PoluInitFile");
	data.output_file	= para.getString("OutputFile");
	data.final_time		= para.getDouble("FinalTime");
	data.anim_jump		= para.getInteger("NbJump");
	data.dirichlet		= para.getDouble("DirichletCondition");
	data.CFL			= para.getDouble("CFL");

	return data;
}

// TODO: convert to cuda
double cpu_compute_mesh_parameter(CFVMesh2D mesh) {
	double h;
	double S;

	h = 1.e20;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		S = mesh.cell_areas[cell];

		for(unsigned int edge = 0; edge < mesh.cell_edges_count[cell]; ++edge) {
			double length = mesh.edge_lengths[edge];
			if (h * length > S)
				h = S / length;
		}
	}

	return h;
}

void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max) {
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		unsigned int left	= mesh.edge_left_cells[i];
		unsigned int right	= mesh.edge_right_cells[i];

		if (right == NO_RIGHT_CELL)
			right = left;

		double v	= ((velocities.x[left] + velocities.x[right]) * 0.5 * mesh.edge_normals.x[i])
					+ ((velocities.y[left] + velocities.y[right]) * 0.5 * mesh.edge_normals.y[i]);

		vs[i] = v;

		if (abs(v) > v_max || i == 0) {
			v_max = abs(v);
		}
		cout << "vs[" << i << "] = " << vs[i] << endl;
	}
}

void cudaSafe(cudaError_t error, const string msg) {
	if (error != cudaSuccess) {
		cerr << "Error: " << msg << " : " << error << endl;
		exit(-1);
	}
}

void cudaCheckError(const string msg) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		cerr << "Error: " << msg << " : " << cudaGetErrorString(error) << endl;
		exit(-1);
	}
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
	FVL::CFVMesh2D mesh(data.mesh_file);

	FVL::CFVPoints2D<double> velocities(mesh.num_cells);
	FVL::CFVArray<double> polution(mesh.num_cells);
	FVL::CFVArray<double> flux(mesh.num_edges);
	FVL::CFVArray<double> vs(mesh.num_edges);
	FVL::CFVMat<double> matA(3, 3, mesh.num_cells);
	FVL::CFVMat<double> vecABC(3, 1, mesh.num_cells);
	FVL::CFVMat<double> vecResult(3, 1, mesh.num_cells);
	FVL::CFVArray<int> vecValidABC(mesh.num_cells);

	// read other input files
	FVL::FVXMLReader velocity_reader(data.velocity_file);
	FVL::FVXMLReader polu_ini_reader(data.initial_file);
	polu_ini_reader.getVec(polution, t, name);
	velocity_reader.getPoints2D(velocities, t, name);
	polu_ini_reader.close();
	velocity_reader.close();

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
	polution.cuda_malloc();
	flux.cuda_malloc();
	vs.cuda_malloc();
	matA.cuda_malloc();
	vecABC.cuda_malloc();
	vecResult.cuda_malloc();
	vecValidABC.cuda_malloc();

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
	cpu_compute_reverseA(
			mesh,
			matA);
	#else
	kernel_compute_reverseA<<< grid_matA, block_matA >>>(
			mesh.cuda_get(),
			matA.cuda_get());

	_D(cudaCheckError("cuda[compute_reverseA]"));

	#endif

	while(t < data.final_time) {
		cout << endl << "iteration " << i << endl;
		/* compute system polution coeficients for system solve */
		#ifdef NO_CUDA
		cpu_compute_vecResult(
					mesh,
					polution,
					vecResult,
					data.dirichlet);
		#else
		kernel_compute_vecResult<<< grid_vecResult, block_vecResult >>>(
				mesh.cuda_get(),
				polution.cuda_get(),
				vecResult.cuda_get(),
				data.dirichlet);

		_DEBUG {
			stringstream ss;
			ss << "cuda[compute_vecResult] i=" << i;
			cudaCheckError(ss.str());
		}
		#endif

		/* compute (a,b,c) vector */
		#ifdef NO_CUDA
		cpu_compute_vecABC(
					mesh,
					matA,
					vecResult,
					vecABC);
		#else
		kernel_compute_vecABC<<< grid_vecABC, block_vecABC >>>(
				mesh.num_cells,
				matA.cuda_get(),
				vecResult.cuda_get(),
				vecABC.cuda_get());

		_DEBUG {
			stringstream ss;
			ss << "cuda[compute_vectABC] i=" << i;
			cudaCheckError(ss.str());
		}
		#endif

		#ifdef NO_CUDA
		cpu_validate_ABC(
					mesh,
					vecABC,
					polution,
					vecValidABC,
					data.dirichlet);
		#else
		kernel_validate_ABC<<< grid_vecValidResult, block_vecValidResult >>>(
					mesh.cuda_get(),
					vs.cuda_get(),
					vecABC.cuda_get(),
					vecValidResult.cuda_get());
		#endif

		/* compute flux */
		#ifdef NO_CUDA
		cpu_compute_flux(
					mesh,
					vs,
					vecABC,
					vecValidABC,
					polution,
					flux,
					data.dirichlet);

		#else
		kernel_compute_flux<<< grid_flux, block_flux >>>(
					mesh.cuda_get(),
					polution.cuda_get(),
					vs.cuda_get(),
					vecABC.cuda_get(),
					flux.cuda_get(),
					data.dirichlet);

		_DEBUG {
			stringstream ss;
			ss << "cuda[compute_flux] i=" << i;
			cudaCheckError(ss.str());
		}
		#endif

		/* update */
		#ifdef NO_CUDA
		cpu_update(
				mesh,
				polution,
				flux,
				dt);
		#else
		kernel_update<<< grid_update, block_update >>>(
				mesh.cuda_get(),
				polution.cuda_get(),
				flux.cuda_get(),
				dt);

		_DEBUG {
			stringstream ss;
			ss << "cuda[update] i=" << i;
			cudaCheckError(ss.str());
		}
		#endif
		
		for(unsigned int x = 0; x < polution.size(); ++x) {
			cout << "polution[ " << x << "] = " << setw(12) << polution[x] << "    { ";
			cout << "a = " << setw(12) << vecABC.elem(0, 0, x) << ", ";
			cout << "b = " << setw(12) << vecABC.elem(1, 0, x) << ", ";
			cout << "c = " << setw(12) << vecABC.elem(2, 0, x) << "}  ";
			cout << "ABCvalid[3] = " << vecValidABC[x] << endl;
		}
		cout << "--------------" << endl;

	t += dt;

	if (i % data.anim_jump == 0) {
		#ifndef NO_CUDA
		polution.cuda_get();
		#endif

		polution_writer.append(polution, t, "polution");
	}

	++i;
}

	// dump final iteration
	#ifndef NO_CUDA
	polution.cuda_get();
	#endif
	polution_writer.append(polution, t, "polution");
	polution_writer.save();
	polution_writer.close();

	#ifndef NO_CUDA
	polution.cuda_free();
	flux.cuda_free();
	vs.cuda_free();
	matA.cuda_free();
	mesh.cuda_free();
	#endif
}

