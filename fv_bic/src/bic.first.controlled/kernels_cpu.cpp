#include "kernels_cpu.h"

inline double _min(double x, double y) {
	return (x < y) ? x : y;
}

inline double _max(double x, double y) {
	return (x > y) ? x : y;
}

/* Aux function for cpu_compute_vecResult - computes ghost cell centroid */
void cpu_ghost_coords(CFVMesh2D &mesh, unsigned int edge, double &x, double &y) {
	// compute lambda
	unsigned int v1 = mesh.edge_fst_vertex[edge];
	unsigned int v2 = mesh.edge_snd_vertex[edge];
	double v1_x = mesh.vertex_coords.x[v1];
	double v2_x = mesh.vertex_coords.x[v2];
	double v1_y = mesh.vertex_coords.y[v1];
	double v2_y = mesh.vertex_coords.y[v2];

	double v1v2_x = v2_x - v1_x;
	double v1v2_y = v2_y - v1_y;

	double lambda	= ((x - v1_x) * v1v2_x	+ (y - v1_y) * v1v2_y)
					/ (v1v2_x * v1v2_x 		+ v1v2_y * v1v2_y);

	// compute AB vector
	double ab_x = x - (v1_x + lambda * v1v2_x);
	double ab_y = y - (v1_y + lambda * v1v2_y);

	// update x & y coords to represent ghost cell
	x -= 2 * ab_x;
	y -= 2 * ab_y;

}

/* compute reverse A matrix kernel */
void cpu_reverseA(CFVMesh2D &mesh, CFVMat<double> &matA) {
	
	for(unsigned int i = 0; i < mesh.num_cells; ++i) {
		// centroid for current cell
		double x0 = mesh.cell_centroids.x[i];
		double y0 = mesh.cell_centroids.y[i];

		matA.elem(0, 0, i) = 0;
		matA.elem(0, 1, i) = 0;
		matA.elem(0, 2, i) = 0;

		matA.elem(1, 0, i) = 0;
		matA.elem(1, 1, i) = 0;
		matA.elem(1, 2, i) = 0;

		matA.elem(2, 0, i) = 0;
		matA.elem(2, 1, i) = 0;
		matA.elem(2, 2, i) = mesh.cell_edges_count[i] + 1;

		// for each edge
		unsigned int edge_limit = mesh.cell_edges_count[i];
		for(unsigned int j = 0; j < edge_limit; ++j) {
			
			// get current edge
			unsigned int edge = mesh.cell_edges.elem(j, 0, i);
			double x, y;
			unsigned int cell_j;

			switch (mesh.edge_types[edge]) {
				// inner edges, both left and right cell can be assumed to exists
				case FV_EDGE:
					// get right cell of this edge
					cell_j = mesh.edge_right_cells[edge];

					// if right cell is the current one (cell i), we want the left instead
					if (cell_j == i)
						cell_j = mesh.edge_left_cells[edge];
					
					// get coords for this cell
					x = mesh.cell_centroids.x[cell_j];
					y = mesh.cell_centroids.y[cell_j];
					break;

				// boundary edges (left_cell == i, there is no right edge, so a ghost cell needs to be used)
				case FV_EDGE_DIRICHLET:
				case FV_EDGE_NEUMMAN:
					// get coords of current cell i, and compute ghost coords
					x = x0;
					y = y0;
					cpu_ghost_coords(mesh, edge, x, y);
					break;
			}

			// system normalization, to place all numbers on the same scale (avoid precision problems with too big numbers against small numbers)
			x -= x0;
			y -= y0;

			// sum to each matrix elem
			matA.elem(0, 0, i) += x * x;
			matA.elem(0, 1, i) += x * y;
			matA.elem(0, 2, i) += x;

			matA.elem(1, 0, i) += x * y;
			matA.elem(1, 1, i) += y * y;
			matA.elem(1, 2, i) += y;

			matA.elem(2, 0, i) += x;
			matA.elem(2, 1, i) += y;
		}

		// A computed, now to the reverse
		
		// determinant
		double det = matA.elem(0, 0, i) *	(matA.elem(1, 1, i) * matA.elem(2, 2, i) -
											 matA.elem(1, 2, i) * matA.elem(2, 1, i))
					- matA.elem(1, 0, i) *	(matA.elem(0, 1, i) * matA.elem(2, 2, i) -
											 matA.elem(0, 2, i) * matA.elem(2, 1, i))
					+ matA.elem(2, 0, i) *	(matA.elem(0, 1, i) * matA.elem(1, 2, i) -
											 matA.elem(0, 2, i) * matA.elem(1, 1, i));

		double invDet = 1.0 / det;

		double tmpA[3][3];
		for(unsigned int x = 0; x < 3; ++x)
			for(unsigned int y = 0; y < 3; ++y)
				tmpA[x][y] = matA.elem(x, y, i);

		matA.elem(0, 0, i) =   (tmpA[1][1] * tmpA[2][2] - tmpA[1][2] * tmpA[2][1]) * invDet;
		matA.elem(0, 1, i) = - (tmpA[1][0] * tmpA[2][2] - tmpA[1][2] * tmpA[2][0]) * invDet;
		matA.elem(0, 2, i) =   (tmpA[1][0] * tmpA[2][1] - tmpA[1][1] * tmpA[2][0]) * invDet;

		matA.elem(1, 0, i) = - (tmpA[0][1] * tmpA[2][2] - tmpA[0][2] * tmpA[2][1]) * invDet;
		matA.elem(1, 1, i) =   (tmpA[0][0] * tmpA[2][2] - tmpA[0][2] * tmpA[2][0]) * invDet;
		matA.elem(1, 2, i) = - (tmpA[0][0] * tmpA[2][1] - tmpA[0][1] * tmpA[2][0]) * invDet;

		matA.elem(2, 0, i) =   (tmpA[0][1] * tmpA[1][2] - tmpA[0][2] * tmpA[1][1]) * invDet;
		matA.elem(2, 1, i) = - (tmpA[0][0] * tmpA[1][2] - tmpA[0][2] * tmpA[1][0]) * invDet;
		matA.elem(2, 2, i) =   (tmpA[0][0] * tmpA[1][1] - tmpA[0][1] * tmpA[1][0]) * invDet;
	}
}

/* Compute vecABC */
void cpu_vecABC(CFVMesh2D &mesh, CFVMat<double> &matA, CFVMat<double> &vecResult, CFVMat<double> &vecABC) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		// A
		vecABC.elem(0, 0, cell) = matA.elem(0, 0, cell) * vecResult.elem(0, 0, cell)
								+ matA.elem(0, 1, cell) * vecResult.elem(1, 0, cell)
								+ matA.elem(0, 2, cell) * vecResult.elem(2, 0, cell);

		// B
		vecABC.elem(1, 0, cell) = matA.elem(1, 0, cell) * vecResult.elem(0, 0, cell)
								+ matA.elem(1, 1, cell) * vecResult.elem(1, 0, cell)
								+ matA.elem(1, 2, cell) * vecResult.elem(2, 0, cell);

		// C
		vecABC.elem(2, 0, cell) = matA.elem(2, 0, cell) * vecResult.elem(0, 0, cell)
								+ matA.elem(2, 1, cell) * vecResult.elem(1, 0, cell)
								+ matA.elem(2, 2, cell) * vecResult.elem(2, 0, cell);
	}
}

/* Compute system polution coeficients for system solve */
void cpu_vecResult(CFVMesh2D &mesh, CFVArray<double> &polution, CFVMat<double> &vecResult, double dc) {
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		double x0 = mesh.cell_centroids.x[cell];
		double y0 = mesh.cell_centroids.y[cell];
		double u = polution[cell];

		// fill initial value of vector
		//vecResult.elem(0, 0, cell) = u * x;
		//vecResult.elem(1, 0, cell) = u * y;
		vecResult.elem(0, 0, cell) = 0;
		vecResult.elem(1, 0, cell) = 0;
		vecResult.elem(2, 0, cell) = u;

		// for each neighbor cell, add to vector
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int j = 0; j < edge_limit; ++j) {

			// get edge
			unsigned int edge = mesh.cell_edges.elem(j, 0, cell);
			double x, y, u;
			unsigned int cell_j;

			switch (mesh.edge_types[edge]) {
				// inner edges, both left and right cell can be assumed to exists
				case FV_EDGE:
					// get right cell of this edge
					cell_j = mesh.edge_right_cells[edge];

					// if right cell is the current one (cell i), we want the left instead
					if (cell_j == cell)
						cell_j = mesh.edge_left_cells[edge];
					
					// get coords for this cell
					x = mesh.cell_centroids.x[cell_j];
					y = mesh.cell_centroids.y[cell_j];
					u = polution[cell_j];
					break;

				// boundary edges (left_cell == i, there is no right edge, so a ghost cell needs to be used)
				case FV_EDGE_DIRICHLET:
					// get left cell
					cell_j = mesh.edge_left_cells[edge];

					// get coords of current cell i, and compute ghost coords
					x = x0;
					y = y0;
					cpu_ghost_coords(mesh, edge, x, y);

					// polution in this point is equal to the dirichlet condition (unless velocity is positive)
					u = dc;

					break;

				case FV_EDGE_NEUMMAN:
					// get left cell
					cell_j = mesh.edge_left_cells[edge];

					// get coords of current cell i, and compute ghost coords
					x = x0;
					y = y0;
					cpu_ghost_coords(mesh, edge, x, y);

					// polution in this ghost cell is equal to the current cell (neumman condition)
					u = polution[cell_j];
					break;
			}

			// system normalization, to place all numbers on the same scale (avoid precision problems with too big numbers against small numbers)
			x -= x0;
			y -= y0;

			// sum to current vec
			vecResult.elem(0, 0, cell) += u * x;
			vecResult.elem(1, 0, cell) += u * y;
			vecResult.elem(2, 0, cell) += u;
		}
	}
}

/* Return result of (A(x-x0) + B(y-y0)) portion of the linear system to compute the flux of an edge */
double cpu_ABC_partial_result(CFVMesh2D &mesh, CFVMat<double> &vecABC, unsigned int edge, unsigned int cell) {

	// get centroid coords for origin cell and edge
	double x = mesh.edge_centroids.x[edge];
	double y = mesh.edge_centroids.y[edge];
	double x0 = mesh.cell_centroids.x[cell];
	double y0 = mesh.cell_centroids.y[cell];

	// compute partial system result
	// Delta(u_i) . B_i . M_ij
	return vecABC.elem(0, 0, cell) * (x - x0) + vecABC.elem(1, 0, cell) * (y + y0);
}

/* Given the values of left edge (u_i), right edge (u_j) and edge value (u_ij) compute Psi value to bound flux between cells */
double cpu_edgePsi(double u_i, double u_j, double u_ij) {
	double ij_minus_i	= u_ij	- u_i;
	double j_minus_i	= u_j	- u_i;
	
	if (ij_minus_i * j_minus_i <= 0) {
		return 0;
	} else {
		return _min(1, j_minus_i / ij_minus_i);
	}
}

/* compute flux kernel */
void cpu_compute_unbounded_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVMat<double> &vecABC, CFVArray<double> &polution,CFVArray<double> &partial_flux, CFVArray<double> &edgePsi, double dc) {

	for(unsigned int edge = 0; edge < mesh.num_edges; ++edge) {
		double v = velocity[edge];
		unsigned int cell_orig, cell_dest;
		double u_i, u_j, u_ij, partial_u_ij, psi;

				
		switch (mesh.edge_types[edge]) {
			case FV_EDGE:
				if (v >= 0) {
					cell_orig = mesh.edge_left_cells[edge];
					cell_dest = mesh.edge_right_cells[edge];
				} else {
					cell_orig = mesh.edge_right_cells[edge];
					cell_dest = mesh.edge_left_cells[edge];
				}

				u_i = polution[cell_orig];
				u_j = polution[cell_dest];

				partial_u_ij = cpu_ABC_partial_result(mesh, vecABC, edge, cell_orig);
				u_ij		 = u_i + partial_u_ij;
				psi			 = cpu_edgePsi(u_i, u_j, u_ij);
				break;

			case FV_EDGE_DIRICHLET:
				cell_orig = mesh.edge_left_cells[edge];
				
				u_i = polution[cell_orig];

				// flux is exiting (positive velocity)
				if (v > 0) {
					// TODO caso em que a velocidade indica que está a sair, isto esta correcto?
					u_j = 0;
					partial_u_ij = cpu_ABC_partial_result(mesh, vecABC, edge, cell_orig);

				// flux is entering (negative velocity)
				} else {
					u_j = u_i;
					u_i	= dc; // TODO caso em que a velocidade indica que está a entrar, isto esta correcto?
					partial_u_ij = dc;
				}

				u_ij		 = partial_u_ij;
				psi = cpu_edgePsi(u_i, u_j, u_ij);
				break;

			case FV_EDGE_NEUMMAN:
				partial_u_ij = 0;
				psi			 = 0;
				break;
		}

		partial_flux[edge]	= partial_u_ij;
		edgePsi[edge]		= psi;

		cout << setw(12)
			<< "partial_flux["	<< edge << "] = " << setw(12) << partial_flux[edge]
			<< "\tedgePsi["		<< edge << "] = " << setw(12) << edgePsi[edge] << endl;
	};
}

/* For each cell, compute min(edgePsi) */
void cpu_cellPsi(CFVMesh2D &mesh, CFVArray<double> &edgePsi, CFVArray<double> &cellPsi) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		double minPsi = 1;
		for(unsigned int edge = 0; edge < mesh.cell_edges_count[cell]; ++edge) {
			double current_edgePsi = edgePsi[ mesh.cell_edges.elem(edge, 0, cell) ];

			if (current_edgePsi < minPsi && mesh.edge_types[edge] != FV_EDGE_NEUMMAN)
				minPsi = current_edgePsi;
		}
		cellPsi[cell] = minPsi;

		cout << setw(12)
			<< "cellPsi[" << cell << "] = " << cellPsi[cell] << endl;
	}
}

/* finalize flux calculation, using cellPsi to bound values */
void cpu_bound_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVArray<double> &cellPsi, CFVArray<double> &polution, CFVArray<double> &flux, double dc) {

	for(unsigned int edge = 0; edge < mesh.num_edges; ++edge) {
		unsigned int cell_orig;
		double v, u_ij, u_i, psi;

		// recover u_i and psi values
		switch(mesh.edge_types[edge]) {
			case FV_EDGE:
				if (v >= 0)
					cell_orig = mesh.edge_left_cells[edge];
				else
					cell_orig = mesh.edge_right_cells[edge];

				u_i = polution[cell_orig];
				psi = cellPsi[cell_orig];
				break;

			case FV_EDGE_DIRICHLET:
				if (v > 0) {
					cell_orig = mesh.edge_left_cells[edge];
					u_i = polution[cell_orig];
					psi = cellPsi[cell_orig];
				} else {
					u_i = dc;
					psi = 1;
				}

				break;

			case FV_EDGE_NEUMMAN:
				u_i = 0;
				psi = 0;
				break;
		}

		// recover u_ij value
		u_ij = flux[edge];

		// recover velocity value for this edge
		v = velocity[edge];

		// compute final flux value, based on u_i, psi, u_ij, and edge velocity
		flux[edge] = v * (u_i + psi * u_ij);
		cout << "flux[" << edge << "] = " << flux[edge] /*<< " (using cell " << cell << ")"*/ << endl;
	}
}

/* update kernel */
void cpu_update(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, double dt) {

	//cout << endl;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int e = 0; e < edge_limit; ++e) {
			unsigned int edge = mesh.cell_edges.elem(e, 0, cell);

			double var = dt * flux[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[cell];

			if (mesh.edge_left_cells[edge] == cell) {
				polution[cell] -= var;
			} else {
				polution[cell] += var;
			}
		}

		//cout << "polution[" << cell << "] = " << polution[cell] << endl;
	}
}
