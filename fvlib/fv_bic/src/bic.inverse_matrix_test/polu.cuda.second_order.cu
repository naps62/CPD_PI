#include <cuda.h>

#include "FVL/CFVLib.h"
#include "FVVect.h"
#include "FVio.h"
#include "Parameter.h"

#ifdef NO_CUDA
#include "kernels_cpu.h"
#else
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
	double comp_threshold;
} Parameters;

// TODO: interface decente para paremetros xml
Parameters read_parameters (string parameters_filename) {
	Parameters data;
	Parameter para(parameters_filename.c_str());

	data.mesh_file		= para.getString("MeshName");
	data.velocity_file	= para.getString("VelocityFile");
	data.initial_file	= para.getString("PoluInitFile");
	data.output_file	= para.getString("OutputFile");
	data.final_time		= para.getDouble("FinalTime");
	data.anim_jump		= para.getInteger("NbJump");
	data.comp_threshold	= para.getDouble("DirichletCondition");

	return data;
}

// TODO: convert to cuda
double compute_mesh_parameter(FVMesh2D mesh) {
	double h;
	double S;
	FVCell2D *cell;
	FVEdge2D *edge;

	h = 1.e20;
	for(mesh.beginCell(); (cell = mesh.nextCell()) != NULL; ) {
		S = cell->area;
		for(cell->beginEdge(); (edge = cell->nextEdge()) != NULL; ) {
			if (h * edge->length > S)
				h = S / edge->length;
		}
	}
	return h;
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

	if (argc != 2) {
		cerr << "Arg error: requires 1 argument (xml param filename)" << endl;
		exit(-1);
	}

	// var declaration
	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	Parameters data = read_parameters(argv[1]);

	// read mesh
	FVL::CFVMesh2D mesh(data.mesh_file);


	// read polution and flux from files
	// TODO: remove this dependency
	FVVect<double> old_polution(mesh.num_cells);
	FVVect<double> old_flux(mesh.num_edges);
	FVVect<FVPoint2D<double> > old_velocity(mesh.num_cells);
	FVio velocity_file(data.velocity_file.c_str(), FVREAD);
	FVio polu_ini_file(data.initial_file.c_str(), FVREAD);
	velocity_file.get(old_velocity, t, name);
	polu_ini_file.get(old_polution, t, name);

	FVL::CFVVect<double> polution(mesh.num_cells);
	FVL::CFVVect<double> flux(mesh.num_edges);
	FVL::CFVVect<double> vs(mesh.num_edges);

	FVL::CFVMat<double> matA(3, 3, mesh.num_cells);
	FVL::CFVVect<double> detA(mesh.num_cells);
	FVL::CFVMat<double> matAReverse(3, 3, mesh.num_cells);
	FVL::CFVMat<double> matAMulRes(3, 3, mesh.num_cells);
	FVL::CFVMat<double> vecABC(3, 1, mesh.num_cells);
	FVL::CFVMat<double> vecResult(3, 1, mesh.num_cells);

	// TODO: remove this dependency
	for(unsigned int i = 0; i < polution.size(); ++i) {
		polution[i] = old_polution[i];
		//vs.x[i] = old_velocity[i].x;
		//vs.y[i] = old_velocity[i].y;
	}

	// compute velocity vector
	// TODO: Convert to CUDA
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		unsigned int left	= mesh.edge_left_cells[i];
		unsigned int right	= mesh.edge_right_cells[i];

		if (right == NO_RIGHT_CELL)
			right = left;

		double v	= ((old_velocity[left].x + old_velocity[right].x) * 0.5 * mesh.edge_normals.x[i])
					+ ((old_velocity[left].y + old_velocity[right].y) * 0.5 * mesh.edge_normals.y[i]);

		vs[i] = v;

		if (abs(v) > v_max || i == 0) {
			v_max = abs(v);
		}

	}

	// TODO: convert to cuda && remove dependency
	FVMesh2D old_mesh;
	old_mesh.read(data.mesh_file.c_str());
	h	= compute_mesh_parameter(old_mesh);
	dt	= 1.0 / v_max * h;

	FVio polution_file(data.output_file.c_str(), FVWRITE);
	polution_file.put(old_polution, t, "polution");

	#ifndef NO_CUDA
	// saves whole mesh to CUDA memory
	mesh.cuda_malloc();
	polution.cuda_malloc();
	flux.cuda_malloc();
	vs.cuda_malloc();
	matA.cuda_malloc();
	detA.cuda_malloc();
	matAReverse.cuda_malloc();
	matAMulRes.cuda_malloc();
	vecABC.cuda_malloc();
	vecResult.cuda_malloc();

	// data copy
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	mesh.vertex_coords.x.cuda_saveAsync(stream);
	mesh.vertex_coords.y.cuda_saveAsync(stream);
	mesh.edge_normals.x.cuda_saveAsync(stream);
	mesh.edge_normals.y.cuda_saveAsync(stream);
	mesh.edge_centroids.x.cuda_saveAsync(stream);
	mesh.edge_centroids.y.cuda_saveAsync(stream);
	mesh.edge_lengths.cuda_saveAsync(stream);
	mesh.edge_fst_vertex.cuda_saveAsync(stream);
	mesh.edge_snd_vertex.cuda_saveAsync(stream);
	mesh.edge_left_cells.cuda_saveAsync(stream);
	mesh.edge_right_cells.cuda_saveAsync(stream);
	mesh.cell_centroids.x.cuda_saveAsync(stream);
	mesh.cell_centroids.y.cuda_saveAsync(stream);
	mesh.cell_perimeters.cuda_saveAsync(stream);
	mesh.cell_areas.cuda_saveAsync(stream);
	mesh.cell_edges_count.cuda_saveAsync(stream);
	for(unsigned int i = 0; i < MAX_EDGES_PER_CELL; ++i) {
		mesh.cell_edges.cuda_saveAsync(stream);
		mesh.cell_edges_normal.cuda_saveAsync(stream);
	}
	polution.cuda_saveAsync(stream);
	vs.cuda_saveAsync(stream);
	

	// sizes of each kernel
	// TODO: mudar BLOCK_SIZE_FLUX para MAT_A
	dim3 grid_matA(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_matA(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_vecResult(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_vecResult(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_vecABC(GRID_SIZE(mesh.num_cells, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_vecABC(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_flux(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_flux(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_update(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_UPDATE), 1, 1);
	dim3 block_update(BLOCK_SIZE_UPDATE, 1, 1);
	#endif

	#ifdef NO_CUDA
	cpu_compute_reverseA(
			mesh,
			matA,
			detA,
			matAReverse);
	#else
	kernel_compute_reverseA<<< grid_matA, block_matA >>>(
			mesh.num_cells,
			mesh.cell_centroids.x.cuda_getArray(),
			mesh.cell_centroids.y.cuda_getArray(),
			mesh.cell_edges_count.cuda_getArray(),
			mesh.cell_edges.cuda_getMat(),
			mesh.edge_left_cells.cuda_getArray(),
			mesh.edge_right_cells.cuda_getArray(),
			matA.cuda_getMat());

	_D(cudaCheckError("cuda[compute_reverseA]"));

	matA.cuda_get();
	detA.cuda_get();
	#endif

	for(unsigned int z = 0; z < mesh.num_cells; ++z) {
		cout << "cell " << z << endl;
		cout << "det: " << detA[z] << endl;
		for(int x = 0; x < 3; ++x) {
			cout << setw(12) << "[";

			for(int y = 0; y < 3; ++y)
				cout << setw(12) << matA.elem(x, y, z) << "   ";
			cout << "] * [ ";

			for(int y = 0; y < 3; ++y)
				cout << setw(12) << matAReverse.elem(x, y, z) << "   ";
			cout << "] = [ ";

			for(int y = 0; y < 3; ++y) {
				double res = 0;
				for(int k = 0; k < 3; ++k)
					res += matA.elem(x, k, z) * matAReverse.elem(k, y, z);
				cout << setw(12) << res << "   ";
			}
			cout << "]" << endl;

		}
		cout << endl;
	}
	exit(0);


	while(t < data.final_time) {
		/* compute system polution coeficients for system solve */
		#ifdef NO_CUDA
		cpu_compute_vecResult(
					mesh,
					polution,
					vecResult);
		#else
		kernel_compute_vecResult<<< grid_vecResult, block_vecResult >>>(
				mesh.num_cells,
				mesh.cell_centroids.x.cuda_getArray(),
				mesh.cell_centroids.y.cuda_getArray(),
				mesh.cell_edges_count.cuda_getArray(),
				mesh.cell_edges.cuda_getMat(),
				mesh.edge_left_cells.cuda_getArray(),
				mesh.edge_right_cells.cuda_getArray(),
				polution.cuda_getArray(),
				vecResult.cuda_getMat());

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
				matA.cuda_getMat(),
				vecResult.cuda_getMat(),
				vecABC.cuda_getMat());

		_DEBUG {
			stringstream ss;
			ss << "cuda[compute_vectABC] i=" << i;
			cudaCheckError(ss.str());
		}
		#endif

 		if (i < 2) {
		#ifndef NO_CUDA
		vecResult.cuda_get();
		vecABC.cuda_get();
		#endif

		cout << "iteration " << i << endl;
		for(unsigned int z = 0; z < mesh.num_cells; ++z) {
			cout << "cell " << z << endl;
			for(int x = 0; x < 3; ++x) {
				cout << setw(12) << vecABC.elem(x, 0, z) << " = [";
				for(int y = 0; y < 3; ++y)
					cout << setw(12) << matA.elem(x, y, z) << "   ";
				cout << "]   [ " << vecResult.elem(x, 0, z) << " ]" << endl;
			}
			cout << endl;
			}
		}
		else
			exit(0);

		/* compute flux */
		#ifdef NO_CUDA
		cpu_compute_flux(
					mesh,
					polution,
					vs,
					vecABC,
					flux,
					data.comp_threshold);

		#else
		kernel_compute_flux<<< grid_flux, block_flux >>>(
					mesh.num_edges,
					mesh.edge_left_cells.cuda_getArray(),
					mesh.edge_right_cells.cuda_getArray(),
					mesh.edge_centroids.x.cuda_getArray(),
					mesh.edge_centroids.y.cuda_getArray(),
					polution.cuda_getArray(),
					vs.cuda_getArray(),
					vecABC.cuda_getMat(),
					flux.cuda_getArray(),
					data.comp_threshold);

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
				mesh.num_cells,
				//mesh.num_total_edges,
				mesh.edge_left_cells.cuda_getArray(),
				mesh.edge_right_cells.cuda_getArray(),
				mesh.edge_lengths.cuda_getArray(),
				mesh.cell_areas.cuda_getArray(),
				mesh.cell_edges.cuda_getMat(),
				//mesh.cell_edges_index.cuda_getArray(),
				mesh.cell_edges_count.cuda_getArray(),
				polution.cuda_getArray(),
				flux.cuda_getArray(),
				dt);

		_DEBUG {
			stringstream ss;
			ss << "cuda[update] i=" << i;
			cudaCheckError(ss.str());
		}
		#endif


	if (i % data.anim_jump == 0) {
		#ifndef NO_CUDA
		polution.cuda_get();
		#endif
		for(unsigned int x = 0; x < mesh.num_cells; ++x)
			old_polution[x] = polution[x];
		
		polution_file.put(old_polution, t, "polution");
	}

	t += dt;
	++i;
}

	// dump final iteration
	#ifndef NO_CUDA
	polution.cuda_get();
	#endif
	for(unsigned int x = 0; x < mesh.num_cells; ++x)
		old_polution[x] = polution[x];

	polution_file.put(old_polution, t, "polution");

	#ifndef NO_CUDA
	polution.cuda_free();
	flux.cuda_free();
	vs.cuda_free();
	matA.cuda_free();
	mesh.cuda_free();
	#endif
}

