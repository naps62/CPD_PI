#include "kernels_cuda.cuh"

#define XX 0
#define YY 1

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


/* Aux function for kernel_compute_vecResult - computes ghost cell centroid */
/*__device__
void cuda_ghost_coords(
		unsigned int *edge_fst_vertex,
		unsigned int *edge_snd_vertex,
		double *vertex_coords_x,
		double *vertex_coords_y,
		unsigned int edge,
		double &x,
		double &y) {

	unsigned int v1 = edge_fst_vertex[edge];
	unsigned int v2 = edge_snd_vertex[edge];

	double v1_x = vertex_coords_x[v1];
	double v2_x = vertex_coords_x[v2];
	double v1_y = vertex_coords_y[v1];
	double v2_y = vertex_coords_y[v2];

	double v1v2_x = v2_x - v1_x;
	double v1v2_y = v2_y - v1_x;

	double lambda	= ((x - v1_x) * v1v2_x	+ (y - v1_y) * v1v2_y)
					/ (v1v2_x * v1v2_x		+ v1v2_y * v1v2_y);

	// compute AB vector
	double ab_x = x - (v1_x + lambda * v1v2_x);
	double ab_y = y - (v1_y + lambda * v1v2_y);

	x -= 2 * ab_x;
	y -= 2 * ab_y;
}*/
__device__ void cuda_ghost_coords(CFVMesh2D_cuda *mesh, unsigned int cell, unsigned int edge, double *res) {
	unsigned int v1 = mesh->edge_fst_vertex[edge];
	unsigned int v2 = mesh->edge_snd_vertex[edge];

	double c1[2], c2[2];
	c1[XX] = mesh->vertex_coords[XX][v1];
	c2[XX] = mesh->vertex_coords[XX][v2];
	c1[YY] = mesh->vertex_coords[YY][v1];
	c2[YY] = mesh->vertex_coords[YY][v2];

	double c1c2[2];
	c1c2[XX] = c2[XX] - c1[XX];
	c1c2[YY] = c2[YY] - c1[YY];

	double x = mesh->cell_centroids[XX][cell];
	double y = mesh->cell_centroids[YY][cell];

	double lambda	= ((x - c1[XX]) * c1c2[XX] + (y - c1[YY]) * c1c2[YY])
					/ (c1c2[XX] * c1c2[XX]     + c1c2[YY] * c1c2[YY]);

	res[XX] = x - (c1[XX] + lambda * c1c2[XX]);
	res[YY] = y - (c1[YY] + lambda * c1c2[YY]);
}

__global__
void cuda_compute_reverseA(CFVMesh2D_cuda *mesh, double **matA) {

	// get thread id
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	double x0 = mesh->cell_centroids[0][cell];
	double y0 = mesh->cell_centroids[1][cell];

	matA[0][cell] = 0;		// elem (0, 0, cell)
	matA[1][cell] = 0;		// elem (1, 0, cell)
	matA[2][cell] = 0;		// elem (2, 0, cell)

	matA[3][cell] = 0;		// elem (0, 1, cell)
	matA[4][cell] = 0;		// elem (1, 1, cell)
	matA[5][cell] = 0;		// elem (2, 1, cell)

	matA[6][cell] = 0;		// elem (0, 2, cell)
	matA[7][cell] = 0;		// elem (1, 2, cell)
	matA[8][cell] = mesh->cell_edges_count[cell] + 1;	// elem (2, 2, cell)

	for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0; --edge_i) {
		// get current edge
		unsigned int edge = mesh->cell_edges[edge_i][cell];
		double coords[2];
		unsigned int cell_j;

		switch(mesh->edge_types[edge]) {
			// inner edges, both left and right cells can be assumed to exist
			case FV_EDGE:
				// get left cell of this edge
				cell_j = mesh->edge_left_cells[edge];

				// if right cell is current one...
				if (cell_j == cell)
					cell_j = mesh->edge_left_cells[edge];

				coords[XX] = mesh->cell_centroids[XX][cell_j];
				coords[YY] = mesh->cell_centroids[YY][cell_j];
				break;

			// boundary edges (left_cell == i, there is no right cell, so a ghost cell needs to be used)
			case FV_EDGE_DIRICHLET:
			case FV_EDGE_FAKE:
			case FV_EDGE_NEUMMAN:
				coords[XX] = x0;
				coords[YY] = y0;

				cuda_ghost_coords(mesh, edge, cell, coords);
				break;
		}

		coords[XX] -= x0;
		coords[YY] -= y0;

		matA[0][cell] += coords[XX] * coords[XX];
		matA[1][cell] += coords[XX] * coords[YY];
		matA[2][cell] += coords[XX];

		matA[3][cell] += coords[XX] * coords[YY];
		matA[4][cell] += coords[YY] * coords[YY];
		matA[5][cell] += coords[YY];

		matA[6][cell] += coords[XX];
		matA[7][cell] += coords[YY];
	}

	__syncthreads();

	double det =	matA[0][cell] * (matA[4][cell] * matA[8][cell] -
									 matA[7][cell] * matA[5][cell])
				-	matA[1][cell] * (matA[3][cell] * matA[8][cell] -
									 matA[6][cell] * matA[5][cell])
				+	matA[2][cell] * (matA[3][cell] * matA[7][cell] -
									 matA[6][cell] * matA[4][cell]);
	double invDet = 1.0 / det;

	double tmpA[9];
	tmpA[0] = (matA[4][cell] * matA[8][cell] - matA[7][cell] * matA[5][cell]) * invDet;
	tmpA[1] = (matA[3][cell] * matA[8][cell] - matA[6][cell] * matA[5][cell]) * invDet;
	tmpA[2] = (matA[3][cell] * matA[7][cell] - matA[6][cell] * matA[4][cell]) * invDet;

	tmpA[3] = (matA[1][cell] * matA[8][cell] - matA[7][cell] * matA[2][cell]) * invDet;
	tmpA[4] = (matA[0][cell] * matA[8][cell] - matA[6][cell] * matA[2][cell]) * invDet;
	tmpA[5] = (matA[0][cell] * matA[7][cell] - matA[6][cell] * matA[1][cell]) * invDet;

	tmpA[6] = (matA[1][cell] * matA[5][cell] - matA[4][cell] * matA[2][cell]) * invDet;
	tmpA[7] = (matA[0][cell] * matA[5][cell] - matA[3][cell] * matA[2][cell]) * invDet;
	tmpA[8] = (matA[0][cell] * matA[4][cell] - matA[3][cell] * matA[1][cell]) * invDet;

	for(unsigned int j = 0; j < 9; ++j)
		matA[j][cell] = tmpA[j];
}

/* Compute vecABC */
__global__ void kernel_compute_vecABC(CFVMesh2D_cuda *mesh, double **matA, double **vecResult, double **vecABC) {

	// get thread id
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	vecABC[0][cell]	= matA[0][cell] * vecResult[0][cell]
					+ matA[3][cell] * vecResult[1][cell]
					+ matA[6][cell] * vecResult[2][cell];

	vecABC[1][cell]	= matA[1][cell] * vecResult[0][cell]
					+ matA[4][cell] * vecResult[1][cell]
					+ matA[7][cell] * vecResult[2][cell];

	vecABC[2][cell]	= matA[2][cell] * vecResult[0][cell]
					+ matA[5][cell] * vecResult[1][cell]
					+ matA[8][cell] * vecResult[2][cell];
}

/* Compute system polution coeficients for system solve */
__global__ void kernel_compute_vecResult(CFVMesh2D_cuda *mesh, double *polution, double **vecResult, double dc) {

	// get thread id
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	double x0 = mesh->cell_centroids[XX][cell];
	double y0 = mesh->cell_centroids[YY][cell];
	double u = polution[cell];

	vecResult[0][cell] = 0;
	vecResult[1][cell] = 0;
	vecResult[2][cell] = u;

	for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0; --edge_i) {
		// get edge
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		double coords[2];
		unsigned int cell_j = mesh->edge_right_cells[edge];

		if (cell_j == cell || cell_j == NO_RIGHT_CELL)
			cell_j = mesh->edge_left_cells[edge];


		switch(mesh->edge_types[edge]) {
			// inner edges, both left and right cell can be assumed to exist
			case FV_EDGE:
				u = polution[cell_j];

				coords[XX] = mesh->cell_centroids[XX][cell_j];
				coords[YY] = mesh->cell_centroids[YY][cell_j];
				break;


			case FV_EDGE_DIRICHLET:
				coords[XX] = x0;
				coords[YY] = y0;
				cuda_ghost_coords(mesh, edge, cell, coords);
				u = dc;
				break;

			case FV_EDGE_FAKE:
			case FV_EDGE_NEUMMAN:
				coords[XX] = x0;
				coords[YY] = y0;
				cuda_ghost_coords(mesh, edge, cell, coords);
				u = polution[cell_j];
				break;
		}

		coords[XX] -= x0;
		coords[YY] -= y0;

		vecResult[0][cell] += u * (coords[XX] - x0);
		vecResult[1][cell] += u * (coords[YY] - y0);
		vecResult[2][cell] += u;
	}
}

__device__ double cuda_ABC_partial_result(CFVMesh2D_cuda *mesh, double **vecABC, unsigned int edge, unsigned int cell, double t, double dt) {
	
	// this part is just for the infinite simulation
	if (mesh->edge_types[edge] == FV_EDGE_FAKE) {
		// check if this edge is actually connected to this cell
		bool correct = false;
		for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0 && !correct; --edge_i)
			if (mesh->cell_edges[edge_i][cell] == edge) correct = true;

		if (!correct) {
			unsigned int left = mesh->edge_left_cells[edge];
			unsigned int right = mesh->edge_right_cells[edge];

			for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0 && !correct; --edge_i) {
				unsigned int new_edge  = mesh->cell_edges[edge_i][cell];

				if (left == mesh->edge_left_cells[new_edge] && right == mesh->edge_right_cells[new_edge]) {
					edge = new_edge;
					correct = true;
				}
			}
		}
	}

	double x = mesh->edge_centroids[XX][edge];
	double y = mesh->edge_centroids[YY][edge];

	double x0 = mesh->cell_centroids[XX][cell];
	double y0 = mesh->cell_centroids[YY][cell];

	// return vecABC[0][cell] * (x-x0) + vecABC[1][cell] * (y-y0);
	return 2*M_PI*cos(2*M_PI*x0 - 2*M_PI*(t + dt/2)) * (x-x0);
}

__global__ void kernel_compute_flux(CFVMesh2D_cuda *mesh, double *polution, double *velocity,double **vecABC, double *flux, double dc, double t, double dt) {

	// get thread id
	unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

	if (edge >= mesh->num_edges) return;

	double v = velocity[edge];
	unsigned int cell_orig, cell_dest;
	double u_i, u_j;
	double partial_u_ij, partial_u_ji;

	switch(mesh->edge_types[edge]) {
		case FV_EDGE:
		case FV_EDGE_FAKE:
			if (v >= 0) {
				cell_orig = mesh->edge_left_cells[edge];
				cell_dest = mesh->edge_right_cells[edge];
			} else {
				cell_orig = mesh->edge_right_cells[edge];
				cell_dest = mesh->edge_left_cells[edge];
			}

			u_i = polution[cell_orig];
			u_j = polution[cell_orig];

			partial_u_ij = cuda_ABC_partial_result(mesh, vecABC, edge, cell_orig, t, dt);
			partial_u_ji = cuda_ABC_partial_result(mesh, vecABC, edge, cell_dest, t, dt);
			break;

		case FV_EDGE_DIRICHLET:
			// TODO correct this one

			cell_orig = mesh->edge_left_cells[edge];

			u_i = polution[cell_orig];

			if (v >= 0) {
				u_j = 0;
				partial_u_ij = cuda_ABC_partial_result(mesh, vecABC, edge, cell_orig, t, dt);
			} else {
				u_j = u_i;
				partial_u_ij = 0;
			}
			break;

		case FV_EDGE_NEUMMAN:
			u_i = 0;
			partial_u_ij = 0;
			break;
	}

	flux[edge] = v * (u_i + partial_u_ij);
}

__global__ void kernel_update(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt) {

	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	// get current polution value for this cell
	double new_polution	= 0;

	// for each edge of this cell
	for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];
		// if this cell is at the left of the edge

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] * mesh->edge_lengths[edge] / mesh->cell_areas[cell];

		// if this cell is on the left or the right of the edge
		if (mesh->edge_left_cells[edge] == cell) {
			new_polution -= aux;
		} else {
			new_polution += aux;
		}
	}

	// update global value
	polution[cell] += new_polution;
}


__global__ void kernel_reset_oldflux(CFVMesh2D_cuda *mesh, double *oldflux) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= mesh->num_cells) return;

	oldflux[tid] = 0;
}


__global__ void kernel_detect_polution_errors(CFVMesh2D_cuda *mesh, double *polution, double *flux, double *oldflux, bool *invalidate_flux) {

	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	double min = mesh->cell_edges[0][cell];
	double max = mesh->cell_edges[0][cell];

	for(int edge_i = mesh->cell_edges_count[cell]; edge_i > 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];

		unsigned int neighbor;

		if (mesh->edge_left_cells[edge] == cell) {
			neighbor = mesh->edge_right_cells[edge];
		} else {
			neighbor = mesh->edge_left_cells[edge];
		}

		if (neighbor != NO_RIGHT_CELL) {
			if (polution[neighbor] < min)
				min = polution[neighbor];
			else if (polution[neighbor] > max)
				max = polution[neighbor];
		}
	}

	double current = polution[cell];
	invalidate_flux[cell] = (current < min || current > max);
}


__global__ void kernel_fix_polution_errors(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double *oldflux, bool *invalidate_flux) {

	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0; --edge_i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];
		unsigned int cell_orig, cell_dest;

		double v = velocity[edge];
		double u_ij, u_ji;

		switch(mesh->edge_types[edge]) {
			case FV_EDGE:
			case FV_EDGE_FAKE:
				if (v >= 0) {
					cell_orig = mesh->edge_left_cells[edge];
					//cell_dest = mesh->edge_right_cells[edge];
				} else {
					cell_orig = mesh->edge_right_cells[edge];
					//cell_dest = mesh->edge_left_cells[edge]
				}

				u_ij = polution[cell_orig];
				u_ji = polution[cell_dest];
				break;

			case FV_EDGE_DIRICHLET:
				cell_orig = mesh->edge_left_cells[edge];
				u_ij = polution[cell_orig];
				break;

			case FV_EDGE_NEUMMAN:
				u_ij = 0;
				break;
		}

		oldflux[edge] = flux[edge];
		flux[edge] = v * u_ij;
	}
}

__global__ void kernel_fix_update(CFVMesh2D_cuda *mesh, double *polution, double *flux, double *oldflux, double dt, bool *invalidate_flux) {

	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell >= mesh->num_cells) return;

	if (invalidate_flux[cell]) {

		for(int edge_i = mesh->cell_edges_count[cell]; edge_i >= 0; --edge_i) {
			unsigned int edge = mesh->cell_edges[edge_i][cell];

			polution[cell] -= dt * (flux[edge] - oldflux[edge]) * mesh->edge_lengths[edge] / mesh->cell_areas[cell];
		}
	}
}