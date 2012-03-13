#include "kernels_cpu.h"

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
void cpu_compute_reverseA(CFVMesh2D &mesh, CFVMat<double> &matA) {
	
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

			// get left cell of this edge
			unsigned int cell_j = mesh.edge_right_cells[edge];
			// if right cell is the current one, or if there is no right cell, use left one
			// TODO: if there is no right edge, is this the right way to create a ghost cell?
			if (cell_j == i || cell_j == NO_RIGHT_CELL)
				cell_j = mesh.edge_left_cells[edge];

			// TODO: was the 2 factor forgotten in the formulas?
			double x = mesh.cell_centroids.x[cell_j];
			double y = mesh.cell_centroids.y[cell_j];

			// if there is no right cell, calc coords of ghost cell
			if (cell_j == i) {
				cpu_ghost_coords(mesh, edge, x, y);
			}

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

			// get right neighbor
			unsigned int neighbor = mesh.edge_right_cells[edge];

			// TODO: ghost cell, used correctly?
			if (neighbor == cell || neighbor == NO_RIGHT_CELL)
				neighbor = mesh.edge_left_cells[edge];
			
			double x = mesh.cell_centroids.x[neighbor];
			double y = mesh.cell_centroids.y[neighbor];
			u = polution[neighbor];

			// if neighbor is still equal to cell, this is a ghost cell, compute centroid)
			if (neighbor == cell) {
				cpu_ghost_coords(mesh, edge, x, y);
			}

			x -= x0;
			y -= y0;

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
		CFVVect<double> &velocity,
		CFVMat<double> &vecABC,
		CFVVect<double> &flux,
		double dc) {
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		double v = velocity[i];
		unsigned int cell;
		double system_res;

		if (v < 0) {
			cell = mesh.edge_right_cells[i];
		} else {
			cell = mesh.edge_left_cells[i];
		}

		if (cell == NO_RIGHT_CELL) {
			// TODO dirichlet(t) = sin(t)
			system_res = dc; // * sin(t)
		} else {
			double x = mesh.edge_centroids.x[i];
			double y = mesh.edge_centroids.y[i];
			double x0 = mesh.cell_centroids.x[cell];
			double y0 = mesh.cell_centroids.y[cell];
			system_res = vecABC.elem(0, 0, cell) * (x - x0) + vecABC.elem(1, 0, cell) * (y - y0) + vecABC.elem(2, 0, cell);
		}

		flux[i] = v * system_res;
	};
}

/* update kernel */
void cpu_update(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVVect<double> &flux,
		double dt) {

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
	}
}
