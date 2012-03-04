#include "kernels_cpu.h"

/* compute reverse A matrix kernel */
void cpu_compute_reverseA(CFVMesh2D &mesh, CFVMat<double> &matA) {
	
	for(unsigned int i = 0; i < mesh.num_cells; ++i) {
		// centroid for current cell
		double x = mesh.cell_centroids.x[i];
		double y = mesh.cell_centroids.y[i];

		matA.elem(0, 0, i) = x * x;
		matA.elem(0, 1, i) = x * y;
		matA.elem(0, 2, i) = x;

		matA.elem(1, 0, i) = x * y;
		matA.elem(1, 1, i) = y * y;
		matA.elem(1, 2, i) = y;

		matA.elem(2, 0, i) = x;
		matA.elem(2, 1, i) = y;
		matA.elem(2, 2, i) = 4;

		// for each edge
		unsigned int edge_limit = mesh.cell_edges_count[i];
		for(unsigned int j = 0; j < edge_limit; ++j) {
			
			// get current edge
			unsigned int edge = mesh.cell_edges.elem(j, 0, i);

			// get left cell of this edge
			unsigned int cell_j = mesh.edge_right_cells[edge];
			// if right cell is the current one, or if there is no right cell, use left one
			// TODO: if there is no right edge, is this the right way to create a ghost cell?
			if (cell_j == i || cell_j == NO_RIGHT_CELL)
				cell_j = mesh.edge_left_cells[edge];

			// TODO: was the 2 factor forgotten in the formulas?
			x = mesh.cell_centroids.x[cell_j];
			y = mesh.cell_centroids.y[cell_j];

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

		double det1 = matA.elem(0, 0, i) *	(matA.elem(1, 1, i) * matA.elem(2, 2, i) -
											 matA.elem(1, 2, i) * matA.elem(2, 1, i));

		double det2	=- matA.elem(1, 0, i) *	(matA.elem(0, 1, i) * matA.elem(2, 2, i) -
											 matA.elem(0, 2, i) * matA.elem(2, 1, i));

		double det3 = matA.elem(2, 0, i) *	(matA.elem(0, 1, i) * matA.elem(1, 2, i) -
											 matA.elem(0, 2, i) * matA.elem(1, 1, i));

		matA.elem(0,0,i) = det1;
		matA.elem(1,0,i) = det2;
		matA.elem(2,0,i) = det3;
		matA.elem(0,1,i) = det;

		matA.elem(0, 0, i) = (tmpA[1][1] * tmpA[2][2] - tmpA[1][2] * tmpA[2][1]) * invDet;
		matA.elem(0, 1, i) = (tmpA[1][0] * tmpA[2][2] - tmpA[1][2] * tmpA[2][0]) * invDet;
		matA.elem(0, 2, i) = (tmpA[1][0] * tmpA[2][1] - tmpA[1][1] * tmpA[2][0]) * invDet;

		matA.elem(1, 0, i) = (tmpA[0][1] * tmpA[2][2] - tmpA[0][2] * tmpA[2][1]) * invDet;
		matA.elem(1, 1, i) = (tmpA[0][0] * tmpA[2][2] - tmpA[0][2] * tmpA[2][0]) * invDet;
		matA.elem(1, 2, i) = (tmpA[0][0] * tmpA[2][1] - tmpA[0][1] * tmpA[2][0]) * invDet;

		matA.elem(2, 0, i) = (tmpA[0][1] * tmpA[1][2] - tmpA[0][2] * tmpA[1][1]) * invDet;
		matA.elem(2, 1, i) = (tmpA[0][0] * tmpA[1][2] - tmpA[0][2] * tmpA[1][0]) * invDet;
		matA.elem(2, 2, i) = (tmpA[0][0] * tmpA[1][1] - tmpA[0][1] * tmpA[1][0]) * invDet;
	}
}

/* Compute vecABC */
void cpu_compute_vecABC(CFVMesh2D &mesh, CFVMat<double> &matA, CFVMat<double> &vecResult, CFVMat<double> &vecABC) {

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
void cpu_compute_vecResult(CFVMesh2D &mesh, CFVVect<double> &polution, CFVMat<double> &vecResult) {
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		double x = mesh.cell_centroids.x[cell];
		double y = mesh.cell_centroids.y[cell];
		double u = polution[cell];

		// fill initial value of vector
		vecResult.elem(0, 0, cell) = u * x;
		vecResult.elem(1, 0, cell) = u * y;
		vecResult.elem(2, 0, cell) = u;

		// for each neighbor cell, add to vector
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int j = 0; j < edge_limit; ++j) {
			// get edge
			unsigned int edge = mesh.cell_edges.elem(j, 0, cell);

			// get right neighbor
			unsigned int neighbor = mesh.edge_right_cells[edge];

			// TODO: ghost cell, used correctly?
			if (neighbor == cell || neighbor == NO_RIGHT_CELL)
				neighbor = mesh.edge_left_cells[edge];

			x = mesh.cell_centroids.x[neighbor];
			y = mesh.cell_centroids.y[neighbor];
			u = polution[neighbor];

			// sum to current vec
			vecResult.elem(0, 0, cell) += u * x;
			vecResult.elem(1, 0, cell) += u * y;
			vecResult.elem(2, 0, cell) += u;
		}
	}
}

/* compute flux kernel */
void cpu_compute_flux(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVVect<double> &velocity,
		CFVVect<double> &flux,
		double dc) {
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		unsigned int i_left		= mesh.edge_left_cells[i];
		unsigned int i_right	= mesh.edge_right_cells[i];

		double p_left, p_right;
		double v;

		p_left	= polution[i_left];
		v		= velocity[i];

		if (i_right != NO_RIGHT_CELL) {
			p_right	= polution[i_right];
		} else {
			p_right = dc;
		}

		if (v < 0)
			flux[i] = v * p_right;
		else
			flux[i] = v * p_left;

	};
}

/* update kernel */
void cpu_update(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVVect<double> &flux,
		double dt) {

	for(unsigned int i = 0; i < mesh.num_cells; ++i) {
		unsigned int edge_limit = mesh.cell_edges_count[i];
		for(unsigned int e = 0; e < edge_limit; ++e) {
			unsigned int edge = mesh.cell_edges.elem(e, 0, i);

			double aux = dt * flux[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[i];

			if (mesh.edge_left_cells[edge] == i) {
				polution[i] -= aux;
			} else {
				polution[i] += aux;
			}
		}
	}
}
