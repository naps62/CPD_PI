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
 * Original compute flux
 */
__global__
void kernel_compute_flux1(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double dc) {
	// thread id = edge index
	unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (edge >= mesh->num_edges) return;

	// velocity of current edge
	double v = velocity[edge];

	unsigned int i_left = mesh->edge_left_cells[edge];
	unsigned int i_right = mesh->edge_right_cells[edge];

	double p_left, p_right;

	p_left	= polution[i_left];
	

	if (i_right != NO_RIGHT_CELL) {
		p_right	 	= polution[i_right];
	} else {
		p_right		= dc;
	}

	/*if (v < 0)
		flux[edge] = v * polution[ mesh->edge_left_cells[edge] ];
	else
		flux[edge] = v * ((mesh->edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh->edge_right_cells[edge] ]);*/
	if (v >= 0)
		flux[edge] = v * p_left;
	else
		flux[edge] = v * p_right;
}

/**
 * Optimization 1 - optimized flux array access
 */
__global__
void kernel_compute_flux2(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double dc) {
	unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

	if (edge >= mesh->num_edges) return;

	double res = velocity[edge];

	unsigned int i_left  = mesh->edge_left_cells[edge];
	unsigned int i_right = mesh->edge_right_cells[edge];

	double p_left, p_right;
	p_left = polution[i_left];

	if (i_right != NO_RIGHT_CELL)
		p_right = polution[i_right];
	else
		p_right = dc;

	if (res >= 0)
		res *= p_left;
	else
		res *= p_right;

	flux[edge] = res;
}

/**
 * Optimization 2 - removed divergence in last cycle
 */
__global__
void kernel_compute_flux2(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double dc) {
	unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

	if (edge >= mesh->num_edges) return;

	double res = velocity[edge];

	unsigned int i_left  = mesh->edge_left_cells[edge];
	unsigned int i_right = mesh->edge_right_cells[edge];

	double p_left, p_right;
	p_left = polution[i_left];

	if (i_right != NO_RIGHT_CELL)
		p_right = polution[i_right];
	else
		p_right = dc;

	/*if (res >= 0)
		res *= p_left;
	else
		res *= p_right;*/
	bool cond = (res >= 0);
	res *= cond * p_left + (!cond) * p_right;

	flux[edge] = res;
}
