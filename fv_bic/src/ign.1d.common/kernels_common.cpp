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


void cpu_compute_a(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &vecA) {

	for(uint cell = 0; cell < mesh.num_cells; ++cell) {
		uint left, right;

		// get left cell
		if (cell == 0) left = mesh.num_cells - 1;
		else           left = cell - 1;

		// get right cell
		if (cell == mesh.num_cells-1) right = 0;
		else                          right = cell + 1;

		// distance between them
		// TODO this is a hack, since all cells have same width (and height=1), then distance is equal to Area * 2
		double dist = mesh.cell_areas[cell] * 2;

		// compute A value
		vecA[cell] = (polution[left] - polution[right]) / dist;

		// cout << "vecA " << cell << " " << vecA[cell] << endl;
	}
}


/* Compute initial u vector */
void cpu_compute_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVArray<double> &vecA) {

	unsigned int left;
	unsigned int right;
	for(int edge = 0; edge < mesh.num_edges; ++edge) {
		// ignore border cells
		if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
			continue;

		left  = mesh.edge_left_cells[edge];
		right = mesh.edge_right_cells[edge];

		recons.u_ij[edge] = polution[left]  + vecA[left]  * mesh.cell_areas[left]  / 2;
		recons.u_ji[edge] = polution[right] + vecA[right] * mesh.cell_areas[right] / 2;

		// cout << "recons ij " << edge << " " << recons.u_ij[edge] << "\t\t\t" << recons.u_ji[edge] << endl;
	}
}