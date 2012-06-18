#include "kernels_cpu.h"

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

				// boundary edges (left_cell == i, there is no right cell, so a ghost cell needs to be used)
				case FV_EDGE_DIRICHLET:
					cout << "this shouldnt happen" << endl;
					break;
				case FV_EDGE_FAKE:
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

#define _USE_MATH_DEFINES
#include <math.h>
/* Compute vecABC */
void cpu_vecABC(CFVMesh2D &mesh, CFVMat<double> &matA, CFVMat<double> &vecResult, CFVMat<double> &vecABC) {

	//cout << endl << endl;
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
				// inner edges, both left and right cell can be assumed to exist
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

				case FV_EDGE_FAKE:
					// get right cell of this edge
					cell_j = mesh.edge_right_cells[edge];

					// if right cell is the current one (cell i), we want the left instead
					if (cell_j == cell)
						cell_j = mesh.edge_left_cells[edge];
					
					// get coords for this cell
					x = x0;
					y = y0;
					cpu_ghost_coords(mesh, edge, x, y);
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
					u = polution[cell_j]; //u = 0; // TODO sera que deveria ser 0 aqui?
					break;
			}

			// system normalization, to place all numbers on the same scale (avoid precision problems with too big numbers against small numbers)
			x -= x0;
			y -= y0;

			// sum to current vec
			vecResult.elem(0, 0, cell) += u * x;
			vecResult.elem(1, 0, cell) += u * y;
			vecResult.elem(2, 0, cell) += u;

			/*cout << endl;
			cout << "cell " << cell << endl;
			for(int x = 0; x < 3; ++x) {
				cout << "\t";
				cout << setw(12) << vecResult.elem(x,0,cell) << endl;
			}*/
		}
	}
}

/* Return result of (A(x-x0) + B(y-y0)) portion of the linear system to compute the flux of an edge */
double cpu_ABC_partial_result(CFVMesh2D &mesh, CFVMat<double> &vecABC, unsigned int edge, unsigned int cell, double t, double dt) {

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

	//return vecABC.elem(0, 0, cell) * (x - x0) + vecABC.elem(1, 0, cell) * (y - y0);
	return 2*M_PI*cos(2*M_PI*x0 - 2*M_PI*(t+dt/2))*(x-x0);
}

bool cpu_edge_horizontal(CFVMesh2D &mesh, int edge) {
	unsigned int v1 = mesh.edge_fst_vertex[edge];
	unsigned int v2 = mesh.edge_snd_vertex[edge];

	return mesh.vertex_coords.y[v1] == mesh.vertex_coords.y[v2];
}

/* compute flux kernel */
void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVMat<double> &vecABC, CFVArray<double> &polution,CFVArray<double> &partial_flux, double dc, double t, double dt) {

	for(unsigned int edge = 0; edge < mesh.num_edges; ++edge) {
		double v = velocity[edge];
		unsigned int cell_orig, cell_dest;
		double u_i, u_j;

		double partial_u_ij, partial_u_ji;
		//double partial_u_ji, partial_u_ij;
		//double u_ij, u_ji;

		switch (mesh.edge_types[edge]) {
			case FV_EDGE:
			case FV_EDGE_FAKE:
				if (v >= 0) {
					cell_orig = mesh.edge_left_cells[edge];
					cell_dest = mesh.edge_right_cells[edge];
				} else {
					cell_orig = mesh.edge_right_cells[edge];
					cell_dest = mesh.edge_left_cells[edge];
				}

				u_i = polution[cell_orig];
				u_j = polution[cell_dest];

				partial_u_ij = cpu_ABC_partial_result(mesh, vecABC, edge, cell_orig, t,dt);
				partial_u_ji = cpu_ABC_partial_result(mesh, vecABC, edge, cell_dest, t,dt);

				break;

			case FV_EDGE_DIRICHLET:
				cell_orig = mesh.edge_left_cells[edge];
				
				u_i = polution[cell_orig];

				// flux is exiting (positive velocity)
				if (v > 0) {
					u_j = 0;
					partial_u_ij = cpu_ABC_partial_result(mesh, vecABC, edge, cell_orig, t,dt);
				// flux is entering (negative velocity)
				} else {
					u_j = u_i;
					u_i	= dc; 
					partial_u_ij = 0;
				}
				break;

			case FV_EDGE_NEUMMAN:
				partial_u_ij = 0;
				break;
		}

		if (cpu_edge_horizontal(mesh, edge))
			partial_flux[edge] = 0;
		else
			partial_flux[edge]	= v * (u_i + partial_u_ij);
	}
}

/* update kernel */
void cpu_update(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, double dt) {

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

void cpu_reset_oldflux(CFVArray<double> &oldflux) {
	for(unsigned int i = 0; i < oldflux.size(); ++i)
		oldflux[i] = 0;
}

void cpu_detect_polution_errors(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, CFVArray<double> &oldflux, CFVArray<bool> &invalidate_flux) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		double min = std::numeric_limits<double>::max();
		double max = std::numeric_limits<double>::min();

		for(unsigned int edge_i = 0; edge_i < mesh.cell_edges_count[cell]; ++edge_i) {
			unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);

			if (mesh.edge_types[edge] != FV_EDGE_NEUMMAN) {

				unsigned int neighbor;

				if (mesh.edge_left_cells[edge] == cell) {
					neighbor = mesh.edge_right_cells[edge];
				} else {
					neighbor = mesh.edge_left_cells[edge];
				}

				if (neighbor != NO_RIGHT_CELL) {
					if (polution[neighbor] < min)
						min = polution[neighbor];
					else if (polution[neighbor] > max)
						max = polution[neighbor];
					else if (edge_i == 0) {
						max = min = polution[neighbor];

					}
				} else cout << "AQUI!" << endl;
			}
		}

		double current = polution[cell];
		invalidate_flux[cell] = (current < min || current > max);
		if (invalidate_flux[cell]) 	cout << "invalidate[" << cell << "]" << " " << min << " " << current << " " << max << endl;
	}
}

void cpu_fix_polution_errors(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &velocity, CFVArray<double> &flux, CFVArray<double> &oldflux, CFVArray<bool> &invalidate_flux) {

	// for each cell that was invalidated
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		if (invalidate_flux[cell]) {

			// calc edges flux based on
			unsigned int edge_limit = mesh.cell_edges_count[cell];
			for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
				unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);
				unsigned int cell_orig;
				double v = velocity[edge];
				double u_ij/*, u_ji*/;

				switch(mesh.edge_types[edge]) {
					case FV_EDGE:
					case FV_EDGE_FAKE:
						if (v >= 0) {
							cell_orig = mesh.edge_left_cells[edge];
							//cell_dest = mesh.edge_right_cells[edge];
						} else {
							cell_orig = mesh.edge_right_cells[edge];
							//cell_dest = mesh.edge_left_cells[edge];
						}

						u_ij = polution[cell_orig];
						//u_ji = polution[cell_orig];
						break;

					case FV_EDGE_DIRICHLET:
						cell_orig = mesh.edge_left_cells[edge];
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
	}
}

void cpu_fix_update(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, CFVArray<double> &oldflux, double dt, CFVArray<bool> &invalidate_flux) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		if (invalidate_flux[cell]) {

			unsigned int edge_limit = mesh.cell_edges_count[cell];
			for(unsigned int e = 0; e < edge_limit; ++e) {
				unsigned int edge = mesh.cell_edges.elem(e, 0, cell);

				polution[cell] -= dt * (flux[edge] - oldflux[edge]) * mesh.edge_lengths[edge] / mesh.cell_areas[cell];
			}
		}
	}
}