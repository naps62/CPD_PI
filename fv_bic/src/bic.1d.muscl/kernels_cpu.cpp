#include "kernels_cpu.h"

#include "../ign.1d.common/kernels_common.cpp"

double _abs(double x) {
	if (x > 0) return x;
	else       return -x;
}

double _signal(double x) {
	if (x >= 0) return 1;
	else        return -1;
}

double cpu_min_mod(double x, double y) {
	if (x*y <= 0)
		return 0;

	return _signal(x) * _min(_abs(x), _abs(y));
}

void cpu_compute_p(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &p) {

	for(uint cell = 0; cell < mesh.num_cells; ++cell) {
		uint left, right;

		// get left cell
		if (cell == 0) left = mesh.num_cells - 1;
		else           left = cell - 1;

		// get right cell
		if (cell == mesh.num_cells-1) right = 0;
		else                          right = cell + 1;

		double p_left  = polution[left];
		double p_this  = polution[cell];
		double p_right = polution[right];

		double dx = mesh.cell_areas[cell];

		// compute P value
		p[cell] = cpu_min_mod((p_right - p_this)/dx,
													(p_this  - p_left)/dx);
	}
}

/* Compute initial u vector */
void cpu_compute_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVArray<double> &p) {

	unsigned int left;
	unsigned int right;
	for(int edge = 0; edge < mesh.num_edges; ++edge) {
		// ignore border cells
		if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
			continue;

		left  = mesh.edge_left_cells[edge];
		right = mesh.edge_right_cells[edge];

		recons.u_ij[edge] = polution[left]  + p[left]  * mesh.cell_areas[left]  / 4; // TODO isto devia ser 2. de onde vem o factor?
		recons.u_ji[edge] = polution[right] - p[right] * mesh.cell_areas[right] / 4;
		//cout << "area: " << mesh.cell_areas[left] << endl;

		// cout << "recons ij " << edge << " " << recons.u_ij[edge] << "\t\t\t" << recons.u_ji[edge] << endl;
	}
}

void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVRecons2D &recons) {

	for(uint edge = 0; edge < mesh.num_edges; ++edge) {
		// ignore border edges
		if (mesh.edge_right_cells[edge] == NO_RIGHT_CELL)
			continue;

		double v = velocity[edge];
		double polu_left, polu_right;

		polu_left  = recons.u_ij[edge];
		polu_right = recons.u_ji[edge];

		if (v >= 0) {
			recons.F_ij[edge] = v * polu_left;
		} else {
			recons.F_ij[edge] = v * polu_right;
		}

		// cout << "recons F_ij " << edge << " "<< v << " " << recons.F_ij[edge] << endl;
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