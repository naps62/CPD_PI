#include "kernels_common.h"

// TODO: convert to cuda
double cpu_compute_mesh_parameter(CFVMesh2D &mesh) {
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

		// TODO better fix for this
		if (mesh.edge_types[i] == FV_EDGE_FAKE)
			vs[i] = 1.0;
	}
}

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
		double x_i = mesh.cell_centroids.x[i];
		double y_i = mesh.cell_centroids.y[i];

		matA.elem(0, 0, i) = 0;
		matA.elem(0, 1, i) = 0;

		matA.elem(1, 0, i) = 0;
		matA.elem(1, 1, i) = 0;

		// for each edge
		unsigned int edge_limit = mesh.cell_edges_count[i];
		for(unsigned int j = 0; j < edge_limit; ++j) {
			
			// get current edge
			unsigned int edge = mesh.cell_edges.elem(j, 0, i);
			double x_j, y_j;
			unsigned int cell_j;

			switch (mesh.edge_types[edge]) {
				// inner edges, both left and right cell can be assumed to exists
				case FV_EDGE:
					// get right cell of this edge
					cell_j = mesh.edge_right_cells[edge];

					// if right cell is the current one (cell i), we want the left instead
					if (cell_j == i)
						cell_j = mesh.edge_left_cells[edge];
					
					// get coords for this x
					x_j = mesh.cell_centroids.x[cell_j];
					y_j = mesh.cell_centroids.y[cell_j];
					break;

				// boundary edges (left_cell == i, there is no right cell, so a ghost cell needs to be used)
				case FV_EDGE_DIRICHLET:
					cout << "this shouldnt happen" << endl;
					break;
				case FV_EDGE_FAKE:
				case FV_EDGE_NEUMMAN:
					// get coords of current cell i, and compute ghost coords
					x_j = x_i;
					y_j = y_i;
					cpu_ghost_coords(mesh, edge, x_j, y_j);
					break;
			}

			// system normalization, to place all numbers on the same scale (avoid precision problems with too big numbers against small numbers)
			double x = x_j - x_i;
			double y = y_j - y_i;

			// sum to each matrix elem
			matA.elem(0, 0, i) += x * x;
			matA.elem(0, 1, i) += x * y;
			matA.elem(1, 0, i) += x * y;
			matA.elem(1, 1, i) += y * y;
		}

		// A computed, now to the reverse
		
		// determinant
		double invDet = 1.0 / (matA.elem(0, 0, i) * matA.elem(1, 1, i) - matA.elem(1, 0, i) * matA.elem(0, 1, i));


		double tmpA[2][2];
		for(unsigned int x = 0; x < 2; ++x)
			for(unsigned int y = 0; y < 2; ++y)
				tmpA[x][y] = matA.elem(x, y, i);

		matA.elem(0, 0, i) =   invDet * tmpA[1][1];
		matA.elem(0, 1, i) = - invDet * tmpA[0][1];
		matA.elem(1, 0, i) = - invDet * tmpA[1][0];
		matA.elem(1, 1, i) =   invDet * tmpA[0][0];
	}
}


/* Compute system polution coeficients for system solve */
void cpu_compute_vecR(CFVMesh2D &mesh, CFVArray<double> &polution, CFVMat<double> &vecR, double dc) {
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		double x_i = mesh.cell_centroids.x[cell];
		double y_i = mesh.cell_centroids.y[cell];
		double u_i = polution[cell];

		// fill initial value of vector
		vecR.elem(0, 0, cell) = 0;
		vecR.elem(1, 0, cell) = 0;

		// for each neighbor cell, add to vector
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int j = 0; j < edge_limit; ++j) {

			// get edge
			unsigned int edge = mesh.cell_edges.elem(j, 0, cell);
			double x_j, y_j, u_j = 0;
			unsigned int cell_j;

			switch (mesh.edge_types[edge]) {
				// inner edges, both left and right cell can be assumed to exist
				case FV_EDGE:
					// get right cell of this edge
					cell_j = mesh.edge_right_cells[edge];

					// if right cell is the current one (cell i), we want the left instead
					if (cell_j == cell)
						cell_j = mesh.edge_left_cells[edge];
					
					// get coords for this cell
					x_j = mesh.cell_centroids.x[cell_j];
					y_j = mesh.cell_centroids.y[cell_j];
					u_j = polution[cell_j];
					break;

				case FV_EDGE_FAKE:
					// get right cell of this edge
					cell_j = mesh.edge_right_cells[edge];

					// if right cell is the current one (cell i), we want the left instead
					if (cell_j == cell)
						cell_j = mesh.edge_left_cells[edge];
					
					// get coords for this cell
					x_j = x_i;
					y_j = y_i;
					cpu_ghost_coords(mesh, edge, x_j, y_j);
					u_j = polution[cell_j];
					break;


				// boundary edges (left_cell == i, there is no right edge, so a ghost cell needs to be used)
				case FV_EDGE_DIRICHLET:
					// get left cell
					cell_j = mesh.edge_left_cells[edge];

					// get coords of current cell i, and compute ghost coords
					x_j = x_i;
					y_j = y_i;
					cpu_ghost_coords(mesh, edge, x_j, y_j);

					// polution in this point is equal to the dirichlet condition (unless velocity is positive)
					u_j = dc;

					break;

				case FV_EDGE_NEUMMAN:
					// get left cell
					cell_j = mesh.edge_left_cells[edge];

					// get coords of current cell i, and compute ghost coords
					x_j = x_i;
					y_j = y_i;
					cpu_ghost_coords(mesh, edge, x_j, y_j);

					// polution in this ghost cell is equal to the current cell (neumman condition)
					u_j = polution[cell_j]; //u = 0; // TODO sera que deveria ser 0 aqui?
					break;
			}

			// sum to current vec
			vecR.elem(0, 0, cell) += (u_j - u_i) * (x_j - x_i);
			vecR.elem(1, 0, cell) += (u_j - u_i) * (y_j - y_i);
		}
	}
}



/* Compute vecABC */
void cpu_compute_gradient(CFVMesh2D &mesh, CFVMat<double> &matA, CFVMat<double> &vecResult, CFVMat<double> &vecGrad) {

	//cout << endl << endl;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		// A
		vecGrad.elem(0, 0, cell) = matA.elem(0, 0, cell) * vecResult.elem(0, 0, cell)
								 + matA.elem(0, 1, cell) * vecResult.elem(1, 0, cell);

		// B
		vecGrad.elem(1, 0, cell) = matA.elem(1, 0, cell) * vecResult.elem(0, 0, cell)
								 + matA.elem(1, 1, cell) * vecResult.elem(1, 0, cell);

		// TODO
		//vecGrad.elem(1, 0, cell) = vecGrad.elem(1, 0, cell) * 2;
	}
}

/* Return result of (A(x-x0) + B(y-y0)) portion of the linear system to compute the flux of an edge */
double cpu_gradient_result(CFVMesh2D &mesh, CFVMat<double> &vecGradient, unsigned int edge, unsigned int cell, double t, double dt) {

	// this part is just for the infinite simulation
	if (mesh.edge_types[edge] == FV_EDGE_FAKE) {
		// check if this edge is actually connected to this cell
		bool correct = false;
		for(unsigned int i = 0; i < mesh.cell_edges_count[cell] && !correct; ++i)
			if (mesh.cell_edges.elem(i, 0, cell) == edge) correct = true;

		// if not, then we're using the fake edge, and should use the right one instead
		//   that is, given the current (fake) edge of the cell, find the corresponding right cell
		if (!correct) {
			unsigned int left  = mesh.edge_left_cells[edge];
			unsigned int right = mesh.edge_right_cells[edge];

			// iterate all real edges of this cell, and find the one connected to the given edge
			for (unsigned int i = 0; i < mesh.cell_edges_count[cell] && !correct; ++i) {
				unsigned int new_edge = mesh.cell_edges.elem(i, 0, cell);
				unsigned int new_left  = mesh.edge_left_cells[new_edge];
				unsigned int new_right = mesh.edge_right_cells[new_edge];

				if (left == new_left && right == new_right) {
					edge = new_edge;
					correct = true;
				}
			}
		}
	}

	// get centroid coords for origin cell and edge
	double x = mesh.edge_centroids.x[edge];
	double y = mesh.edge_centroids.y[edge];
	double x0 = mesh.cell_centroids.x[cell];
	double y0 = mesh.cell_centroids.y[cell];


	//cout << cell << endl;
	//cout << x0 << " grad a " << vecGradient.elem(0,0,cell) << " " << (vecGradient.elem(0,0,cell) - 2*M_PI*cos(2*M_PI*(x0))*sin(2*M_PI*y0)) << endl;
	//cout << y0 << " grad b " << vecGradient.elem(1,0,cell) << " " << (vecGradient.elem(1,0,cell) - 2*M_PI*cos(2*M_PI*(y0))*sin(2*M_PI*x0)) << endl;
	//cout << endl;

	return vecGradient.elem(0, 0, cell) * (x - x0) + vecGradient.elem(1, 0, cell) * (y - y0);
	//return 2*M_PI*cos(2*M_PI*x0 - 2*M_PI*(t+dt/2))*(x-x0);
}


/* Compute initial u vector */
void cpu_compute_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVMat<double> &vecGradient, double t, double dt) {

	unsigned int cell_i;
	unsigned int cell_j;

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		cell_i = mesh.edge_left_cells[edge];
		cell_j = mesh.edge_right_cells[edge];
		recons.u_ij[edge] = polution[cell_i] + cpu_gradient_result(mesh, vecGradient, edge, cell_i, t, dt);

		if (cell_j != NO_RIGHT_CELL) {
			recons.u_ji[edge] = polution[cell_j] + cpu_gradient_result(mesh, vecGradient, edge, cell_j, t, dt);
		}
	}
}

void cpu_compute_border_u(CFVMesh2D &mesh, CFVRecons2D &recons, double dc) {

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		// TODO is this correct?
		switch(mesh.edge_types[edge]) {
			case FV_EDGE_DIRICHLET:	recons.u_ji[edge] = 0;  break;
			case FV_EDGE_NEUMMAN:	recons.u_ji[edge] = dc; break;
		}
	}
		//if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
		//	recons.u_ji[edge] = dc; // TODO is this correct?
}


/*bool cpu_edge_horizontal(CFVMesh2D &mesh, int edge) {
	unsigned int v1 = mesh.edge_fst_vertex[edge];
	unsigned int v2 = mesh.edge_snd_vertex[edge];

	return mesh.vertex_coords.y[v1] == mesh.vertex_coords.y[v2];
}*/