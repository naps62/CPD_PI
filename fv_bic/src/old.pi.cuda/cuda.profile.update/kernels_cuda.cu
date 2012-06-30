#include "kernels_cuda.cuh"

__host__ void cudaSafe(cudaError_t error, const string msg) {
	if (error != cudaSuccess) {
		cerr << "Error: " << msg << " : " << error << endl;
		exit(-1);
	}
}

__host__ void cudaCheckError(const string msg) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		cerr << "Error: " << msg << " : " << cudaGetErrorString(error) << endl;
		exit(-1);
	}
}

// TODO: convert to cudaa
__host__  double kernel_compute_mesh_parameter(CFVMesh2D &mesh) {
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

__host__ void kernel_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max) {
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
	}
}

/**
 * Original update
 */
__global__
void kernel_update1(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= mesh->num_cells) return;

	// define start and end of neighbor edges
	unsigned int edge_limit = mesh->cell_edges_count[cell];

	// for each edge of this cell
	for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];
		// if this cell is at the left of the edge

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] *
			mesh->edge_lengths[edge] /
			mesh->cell_areas[cell];

		// if this cell is on the left or the right of the edge
		if (mesh->edge_left_cells[edge] == cell) {
			polution[cell] -= aux;
		} else {
			polution[cell] += aux;
		}
	}
}


/**
 * Optimization 1 -- optimize polution[cell] memory acess
 */
__global__
void kernel_update2(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= mesh->num_cells) return;

	// define start and end of neighbor edges
	unsigned int edge_limit = mesh->cell_edges_count[cell];

	// get current polution value for this cell
	double new_polution	= polution[cell];

	// for each edge of this cell
	for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] *
			mesh->edge_lengths[edge] /
			mesh->cell_areas[cell];

		// if this cell is on the left or the right of the edge
		if (mesh->edge_left_cells[edge] == cell) {
			new_polution -= aux;
		} else {
			new_polution += aux;
		}
	}

	polution[cell] = new_polution;
}

/**
 * Optimization 2 -- changed loop direction
 */
__global__
void kernel_update3(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= mesh->num_cells) return;

	// get current polution value for this cell
	double new_polution	= polution[cell];

	for(int edge_i = mesh->cell_edges_count[cell] - 1; edge_i >= 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] *
			mesh->edge_lengths[edge] /
			mesh->cell_areas[cell];

		// if this cell is on the left or the right of the edge
		if (mesh->edge_left_cells[edge] == cell) {
			new_polution -= aux;
		} else {
			new_polution += aux;
		}
	}

	polution[cell] = new_polution;
}

/**
 * Optimization 3 -- remove divergence
 */
__global__
void kernel_update4(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= mesh->num_cells) return;

	// get current polution value for this cell
	double new_polution	= polution[cell];

	// for each edge of this cell
	for(int edge_i = mesh->cell_edges_count[cell] - 1; edge_i >= 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] *
			mesh->edge_lengths[edge] /
			mesh->cell_areas[cell];

		// if this cell is on the left or the right of the edge
		new_polution += aux * (2*(int)(mesh->edge_left_cells[edge] == cell) - 1);
	}

	polution[cell] = new_polution;
}

/**
 * Optimization 4 -- optimized out division
 */
__global__
void kernel_update5(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt, double **length_area_ratio) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= mesh->num_cells) return;

	// get current polution value for this cell
	double new_polution	= polution[cell];

	// for each edge of this cell
	for(int edge_i = mesh->cell_edges_count[cell] - 1; edge_i >= 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] * (double) length_area_ratio[edge_i][cell];

		// if this cell is on the left or the right of the edge
		new_polution += aux * (2*(int)(mesh->edge_left_cells[edge] == cell) - 1);
	}

	polution[cell] = new_polution;
}

/**
 * Optimization 5 -- added syncthreads
 */
__global__
void kernel_update6(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt, double **length_area_ratio) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= mesh->num_cells) return;

	// get current polution value for this cell
	double new_polution	= polution[cell];

	// for each edge of this cell
	for(int edge_i = mesh->cell_edges_count[cell] - 1; edge_i >= 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] * (double) length_area_ratio[edge_i][cell];

		// if this cell is on the left or the right of the edge
		new_polution += aux * (2*(int)(mesh->edge_left_cells[edge] == cell) - 1);
	}

	__syncthreads();
	polution[cell] = new_polution;
}