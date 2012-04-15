#include "kernels.h"

/* compute flux kernel */
void compute_flux(FVMesh2D_SOA &mesh, CFVArray<double> &velocity, CFVArray<double> &polution, CFVArray<double> &flux, double dc) {
	// TODO
}

/* update kernel */
void update(FVMesh2D_SOA &mesh, CFVArray<double> &polution, CFVArray<double> &flux, double dt) {

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
