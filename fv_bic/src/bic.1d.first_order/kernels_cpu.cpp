#include "kernels_cpu.h"

/* compute flux kernel */
void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVArray<double> &polution, CFVArray<double> &flux, double dc) {

	for(unsigned int edge = 0; edge < mesh.num_edges; ++edge) {

		if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
			continue;

		double v = velocity[edge];
		double polu_left, polu_right;
		
		polu_left	 = polution[ mesh.edge_left_cells[edge] ];
		polu_right = polution[ mesh.edge_right_cells[edge] ];

		if (v >= 0)
			flux[edge] = v * polu_left;
		else
			flux[edge] = v * polu_right;


		// cout << "recons F_ij " << edge << " " << v<< flux[edge] << endl;
	};
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
