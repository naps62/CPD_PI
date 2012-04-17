#include "kernels.h"

/* compute flux kernel */
void compute_flux(FVMesh2D_SOA &mesh, FVArray<double> &velocity, CFVArray<double> &polution, FVArray<double> &flux, double dc) {
	for(unsigned edge = 0; edge < mesh.num_edges; ++edge) {
		double polu_left, polu_right;
		double v = velocity[edge];

		polu_left	= polution[ mesh.edge_left_cells[edge] ];
		polu_right	= (mesh.edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh.edge_right_cells[edge] ];


		if (v >= 0)
			flux[edge] = v * polu_left;
		else
			flux[edge] = v * polu_right;

		/*
		polu_right	= (mesh.edge_right_cells[edge] == NO_RIGHT_CELL) * dc
					+ (mesh.edge_right_cells[edge] != NO_RIGHT_CELL) * polution[ mesh.edge_right_cells[edge] ];

		flux[edge]	= (v >= 0) * v * polu_left
					+ (v <  0) * v * polu_right;
		*/
	}
}

/* update kernel */
void update(FVMesh2D_SOA &mesh, FVArray<double> &polution, FVArray<double> &flux, double dt) {

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
