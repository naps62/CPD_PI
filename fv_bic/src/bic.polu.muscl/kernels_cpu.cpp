#include "kernels_cpu.h"

#include "../ign.polu.common/kernels_common.cpp"

/* Given the values of left edge (u_i), right edge (u_j) and edge value (u_ij) compute Psi value to bound flux between cells */
double cpu_edgePsi(double u_i, double u_j, double u_ij, double u_ji) {
	double ij_minus_i	= u_ij	- u_i;
	double j_minus_ji   = u_j   - u_ji;
	double j_minus_i	= u_j	- u_i;
	
	if (ij_minus_i * j_minus_i <= 0 || j_minus_ji * j_minus_i <= 0) {
		return 0;
	} else {
		//cout << "_min(1, " << j_minus_i << " / " << ij_minus_i << ")" << endl;
		//cout << (ij_minus_i) << " " << j_minus_i << " " << ij_minus_i / j_minus_i <<  endl;
		return _min(1, _min(ij_minus_i / j_minus_i, j_minus_ji / j_minus_i));
	}
}

//void cpu_compute_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity) {
void cpu_compute_unbounded_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &vs, CFVArray<double> &polution, CFVArray<double> &edgePsi, double dc) {

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		double v = vs[edge];

		if (v >= 0)	recons.F_ij[edge] = v * recons.u_ij[edge];
		else		recons.F_ij[edge] = v * recons.u_ji[edge];

		unsigned int l, r;
		l = mesh.edge_left_cells[edge];
		r = mesh.edge_right_cells[edge];

		double u_i, u_j;
		u_i = polution[l];
		switch(mesh.edge_types[edge]) {
			case FV_EDGE:
			case FV_EDGE_FAKE:      u_j = polution[r]; break;
			case FV_EDGE_DIRICHLET: u_j = dc;          break;
			case FV_EDGE_NEUMMAN:   u_j = 0;           break;
		}
		edgePsi[edge] = cpu_edgePsi(u_i, u_j, recons.u_ij[edge], recons.u_ji[edge]);
	}
}

/* For each cell, compute min(edgePsi) */
void cpu_cellPsi(CFVMesh2D &mesh, CFVArray<double> &edgePsi, CFVArray<double> &cellPsi) {
	//cout << endl;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		double minPsi = 1;
		for(unsigned int edge_i = 0; edge_i < mesh.cell_edges_count[cell]; ++edge_i) {
			unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);

			double current_edgePsi = edgePsi[ edge ];
			//cout << "edge " << edge << " edgePsi " << current_edgePsi << endl;

			//cout << "edge " << mesh.cell_edges.elem(edge,0,cell) << " psi " << current_edgePsi << endl;
			if (current_edgePsi < minPsi && mesh.edge_types[edge] != FV_EDGE_NEUMMAN /*&&

				// TODO this prevents horizontal edges from being used. not a pretty solution
				mesh.vertex_coords.y[ mesh.edge_fst_vertex[edge] ] != mesh.vertex_coords.y[ mesh.edge_snd_vertex[edge] ]*/) {
				minPsi = current_edgePsi;
			}
				
		}
		cellPsi[cell] = minPsi;
		
		//cellPsi[cell] = 1;
	}
}

/* Compute initial u vector */
void cpu_bound_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVMat<double> &vecGradient, CFVArray<double> &cellPsi, double t, double dt) {

	unsigned int cell_i;
	unsigned int cell_j;

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		cell_i = mesh.edge_left_cells[edge];
		cell_j = mesh.edge_right_cells[edge];
		//cout << "cellPsi " << cell_i << " " << cellPsi[cell_i];
		recons.u_ij[edge] = polution[cell_i] + cellPsi[cell_i] * cpu_gradient_result(mesh, vecGradient, edge, cell_i, t, dt);

		if (cell_j != NO_RIGHT_CELL) {
			//cout << " " << cell_j << " " << cellPsi[cell_j];
			recons.u_ji[edge] = polution[cell_j] + cellPsi[cell_j] * cpu_gradient_result(mesh, vecGradient, edge, cell_j, t, dt);
		}
		//cout << endl;
	}
}

//void cpu_compute_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity) {
void cpu_bound_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &vs, CFVArray<double> &polution, double dc) {

	for(int edge = mesh.num_edges - 1; edge >= 0; --edge) {
		double v = vs[edge];

		if (v >= 0)	recons.F_ij[edge] = v * recons.u_ij[edge];
		else		recons.F_ij[edge] = v * recons.u_ji[edge];
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