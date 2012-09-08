#include "kernels_cpu.h"

#include "../ign.polu.common/kernels_common.cpp"

void cpu_compute_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity) {

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		double v = velocity[edge];

		if (v >= 0)	recons.F_ij[edge] = v * recons.u_ij[edge];
		else		recons.F_ij[edge] = v * recons.u_ji[edge];

		recons.F_ij_old[edge] = 0;
		recons.edge_state[edge] = true;
	}
}

/* update kernel */
void cpu_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		double initial = polution[cell];

		for(unsigned int e = 0; e < edge_limit; ++e) {
			unsigned int edge = mesh.cell_edges.elem(e, 0, cell);

			//cout << edge << " flux " << recons.F_ij[edge] << endl;
			double var = dt * recons.F_ij[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[cell];

			if (mesh.edge_left_cells[edge] == cell)
				polution[cell] -= var;
			else
				polution[cell] += var;
		}

		//cout << cell << " polution from " << initial << " to " << polution[cell] << endl;
		//cout << "dt " << dt << endl << endl;

		recons.degree[cell] = 1;
	}
}

/* invalidate an edge, and consequentely both of its neighbor cells */
void cpu_invalidate_edge(CFVMesh2D &mesh, CFVRecons2D &recons, unsigned int edge) {
	recons.edge_state[edge] = false;

	unsigned int l = mesh.edge_left_cells[edge];
	unsigned int r = mesh.edge_right_cells[edge];

	recons.cell_state[l] = false;
	if (r != NO_RIGHT_CELL) recons.cell_state[r] = false;
}

/* detect bad cells */
bool cpu_bad_cell_detector(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution) {

	bool result = false;
	//int count = 0;

	for(int cell = mesh.num_cells - 1; cell >= 0; --cell) {

		if (recons.degree[cell] == 0) {
			continue;
		}

		double current = polution[cell];
		int edge_start = mesh.cell_edges_count[cell] - 1;
		double min = std::numeric_limits<double>::max();
		double max = std::numeric_limits<double>::min();

		bool first = true;
		for(int edge_i = edge_start; edge_i >= 0; --edge_i) {
			unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);

			if (mesh.edge_types[edge] != FV_EDGE_NEUMMAN) {
				int neighbor = mesh.edge_left_cells[edge];

				if (neighbor == cell)
					neighbor = mesh.edge_right_cells[edge];

				if (neighbor != NO_RIGHT_CELL) {
					double u = polution[neighbor];
					if (first) {
						min = max = u;
						first = false;
					}
					else {
						min = _min(u, min);
						max = _max(u, max);
					}
				}
			}
		}
		//cout << cell << " " << min << " " << max << endl;

		// if current cell is invalid, declare all of its edges as invalid
		recons.cell_state[cell] = (current >= min && current <= max);
		//if (recons.degree[cell] == 1)
		//	recons.cell_state[cell] = false;
		//if (recons.cell_state[cell] == 0)
		//	cout << cell << " " << recons.cell_state[cell] << " " << min << " " << current << " " << max << endl;

		if (recons.cell_state[cell] == false) {
			recons.degree[cell] = 0;
			result = true;
			
			for(int edge_i = edge_start; edge_i >= 0; --edge_i) {
				unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);
				cpu_invalidate_edge(mesh, recons, edge);
			}
		}

	}	
	
	return result;
}

void cpu_fix_u(CFVMesh2D &mesh,CFVRecons2D &recons, CFVArray<double> &polution) {

	unsigned int cell_i;
	unsigned int cell_j;

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		// if edge needs to be fixed
		if (recons.edge_state[edge] == false) {
			cell_i = mesh.edge_left_cells[edge];
			cell_j = mesh.edge_right_cells[edge];

			recons.u_ij[edge] = polution[cell_i];

			if (cell_j != NO_RIGHT_CELL) {
				recons.u_ji[edge] = polution[cell_j];
			}
		}
	}
}

// TODO se a cpu_compute_border_u estiver correcta, esta nao é necessária, pois é redundante
void cpu_fix_border_u(CFVMesh2D &mesh, CFVRecons2D &recons, double dc) {

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge)
		if (recons.edge_state[edge] == false && mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
			recons.u_ji[edge] = dc;
}

void cpu_fix_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity) {

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		//unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);/cout
		recons.F_ij_old[edge] = recons.F_ij[edge];

//		if (recons.edge_state[edge] == false) {
		double v = velocity[edge];

		if (v >= 0)	recons.F_ij[edge] = v * recons.u_ij[edge];
		else		   recons.F_ij[edge] = v * recons.u_ji[edge];

//			recons.edge_state[edge] = true;
//		}
	}
}

void cpu_fix_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &candidate_polution, CFVArray<double> &polution, double dt) {

	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		if (recons.cell_state[cell] == false) {
			
			//double initial = polution[cell];
			candidate_polution[cell] = polution[cell];
			unsigned int edge_limit = mesh.cell_edges_count[cell];
			double var = 0;
			for(unsigned int e = 0; e < edge_limit; ++e) {
				unsigned int edge = mesh.cell_edges.elem(e, 0, cell);

				//if (cell == 11) cout << endl << " flux" << recons.F_ij[edge] << " old " << recons.F_ij_old[edge];
			
				//cout << edge << " flux: old " << recons.F_ij_old[edge] << " new " << recons.F_ij[edge] << endl;
				//cout << cell << " pol: from " << polution[cell] << " to ";
				var = dt * (recons.F_ij[edge]/* - recons.F_ij_old[edge]*/) * mesh.edge_lengths[edge] / mesh.cell_areas[cell];
				//cout << polution[cell] << endl;

				if (mesh.edge_left_cells[edge] == cell)
					candidate_polution[cell] -= var;
				else
					candidate_polution[cell] += var;
			}

			//cout << cell << " from " << initial <<  " to " << polution[cell] << endl << endl;

			recons.cell_state[cell] = true;
		}
	}
}