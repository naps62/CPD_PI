#include <cuda.h>

#include "FVL/CFVLib.h"
#include "FVVect.h"
#include "FVio.h"
#include "Parameter.h"

#include "kernels.cuh"

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

int main(int argc, char **argv) {
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

		if (right == NO_RIGHT_EDGE)
			right = left;

		double v	= ((old_velocity[left].x + old_velocity[right].x) * 0.5 * mesh.edge_normals.x[i])
					+ ((old_velocity[left].y + old_velocity[right].y) * 0.5 * mesh.edge_normals.y[i]);

		vs[i] = v;

		if (abs(v) > v_max || i == 0)
			v_max = abs(v);

	}

	// TODO: convert to cuda && remove dependency
	FVMesh2D old_mesh;
	old_mesh.read(data.mesh_file.c_str());
	h	= compute_mesh_parameter(old_mesh);
	dt	= 1.0 / v_max * h;

	FVio polution_file(data.output_file.c_str(), FVWRITE);
	polution_file.put(old_polution, t, "polution");

	// saves whole mesh to CUDA memory
	mesh.cuda_malloc();
	polution.cuda_malloc();
	flux.cuda_malloc();
	vs.cuda_malloc();

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
	mesh.edge_right_cells.cuda_saveAsync(stream);
	mesh.cell_centroids.x.cuda_saveAsync(stream);
	mesh.cell_centroids.y.cuda_saveAsync(stream);
	mesh.cell_perimeters.cuda_saveAsync(stream);
	mesh.cell_areas.cuda_saveAsync(stream);
	mesh.cell_edges_count.cuda_saveAsync(stream);
	for(unsigned int i = 0; i < MAX_EDGES_PER_CELL; ++i) {
		mesh.cell_edges.cuda_saveAsync(stream);
		mesh.cell_edges_normal.cuda_saveAsync(stream);
		mesh.cell_edges_normal.cuda_saveAsync(stream);
	}

	// sizes of each kernel
	dim3 grid_flux(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_FLUX), 1, 1);
	dim3 block_flux(BLOCK_SIZE_FLUX, 1, 1);

	dim3 grid_update(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_UPDATE), 1, 1);
	dim3 block_update(BLOCK_SIZE_UPDATE, 1, 1);

	while(t < data.final_time) {
		/* compute flux */
#ifdef NO_CUDA
		kernel_compute_flux(
					mesh,
					polution,
					vs,
					flux,
					data.comp_threshold);
#else
		kernel_compute_flux<<< grid_flux, block_flux >>>(
					mesh.num_edges,
					mesh.edge_left_cells.cuda_getArray(),
					mesh.edge_right_cells.cuda_getArray(),
					polution.cuda_getArray(),
					vs.cuda_getArray(),
					flux.cuda_getArray(),
					data.comp_threshold);
#endif

		flux.cuda_get();
		for(int x = 0; x < 10; ++x) {
			cout << flux[x] << "\n";
		}
		exit(0);

		/* update */
#ifdef NO_CUDA
		kernel_update(
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

	polution.cuda_free();
	flux.cuda_free();
	vs.cuda_free();
	mesh.cuda_free();
}

