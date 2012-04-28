#include "kernels.h"

#include <limits>


/* compute flux kernel */
void compute_flux(FVMesh2D_SOA &mesh, FVArray<double> &velocity, CFVArray<double> &polution, FVArray<double> &flux, double dc) {
	for(unsigned edge = 0; edge < mesh.num_edges; ++edge) {
		double polu_left, polu_right;
		double v = velocity[edge];

		polu_left	= polution[ mesh.edge_left_cells[edge] ];
		polu_right	= (mesh.edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh.edge_right_cells[edge] ];

		if (v >= 0) flux[edge] = v * polu_left;
		else		flux[edge] = v * polu_right;

		/*
		if (v >= 0) flux[edge] = polution[ mesh.edge_left_cells[edge] ];
		else		flux[edge] = ((mesh.edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh.edge_right_cells[edge] ]);
		flux[edge] *= v;
		*/

		/*
		   polu_left	= polution[ mesh.edge_left_cells[edge] ];
		   polu_right	= _choose(mesh.edge_right_cells[edge] == NO_RIGHT_CELL, dc, polution[ mesh.edge_right_cells[edge] ]);
		   flux		= _choose(v >= 0, v * polu_left, v * polu_right)
		   */
	}
}

/* update kernel */
void update(FVMesh2D_SOA &mesh, FVArray<double> &polution, FVArray<double> &flux, double dt) {

	//cout << endl;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
			unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);

			double var = dt * flux[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[cell];

			if (mesh.edge_left_cells[edge] == cell) {
				polution[cell] -= var;
			} else {
				polution[cell] += var;
			}

			/*
			polution[cell] += _choose(mesh.edge_left_cells[edge] == cell, -var, var);
			*/
		}
	}
}
