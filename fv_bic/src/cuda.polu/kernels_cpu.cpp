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
	}
}

void cpu_compute_length_area_ratio(CFVMesh2D &mesh, CFVMat<double> &length_area_ratio) {
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {

		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
			unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);

			length_area_ratio.elem(edge_i, 0, cell) = mesh.edge_lengths[edge] / mesh.cell_areas[cell];
		}
	}
}

/* compute flux kernel */
void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVArray<double> &polution,CFVArray<double> &flux, double dc) {

	for(unsigned int edge = 0; edge < mesh.num_edges; ++edge) {
		double v = velocity[edge];
		double polu_left, polu_right;
		
		polu_left	= polution[ mesh.edge_left_cells[edge] ];
		polu_right	= (mesh.edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh.edge_right_cells[edge] ];

		if (v >= 0)
			flux[edge] = v * polu_left;
		else
			flux[edge] = v * polu_right;
	};
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

void cpu_update_optim(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, double dt, CFVMat<double> &length_area_ratio) {
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int e = 0; e < edge_limit; ++e) {
			unsigned int edge = mesh.cell_edges.elem(e, 0, cell);

			double var = dt * flux[edge] * length_area_ratio.elem(e, 0, cell);

			if (mesh.edge_left_cells[edge] == cell) {
				polution[cell] -= var;
			} else {
				polution[cell] += var;
			}
		}
	}
}