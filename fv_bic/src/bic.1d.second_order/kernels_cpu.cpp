#include "kernels_cpu.h"

#include "../ign.1d.common/kernels_common.cpp"

void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &vs, CFVRecons2D &recons) {

	for(uint edge = 0; edge < mesh.num_edges; ++edge) {
		// ignore border edges
		if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
			continue;

		double v = vs[edge];
		double polu_left, polu_right;

		polu_left  = recons.u_ij[edge];
		polu_right = recons.u_ji[edge];

		if (v >= 0) {
			recons.F_ij[edge] = v * polu_left;
		} else {
			recons.F_ij[edge] = v * polu_right;
		}
	}
}

/* update kernel */
void cpu_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		unsigned int edge_limit = mesh.cell_edges_count[cell];

		for(unsigned int e = 0; e < edge_limit; ++e) {
			unsigned int edge = mesh.cell_edges.elem(e, 0, cell);

			// ignore border edges
			if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
				continue;

			//cout << edge << " flux " << recons.F_ij[edge] << endl;
			double var = dt * recons.F_ij[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[cell];

			if (mesh.edge_left_cells[edge] == cell)
				polution[cell] -= var;
			else
				polution[cell] += var;
		}
	}
}