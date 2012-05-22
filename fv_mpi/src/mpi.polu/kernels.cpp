#include "kernels.h"

#include <mpi.h>
#include <limits>

enum TAGS {
	TAG_LEFT_COMM,
	TAG_RIGHT_COMM,
};

/* communication step */
void communication(int id, int size, FVMesh2D_SOA_Lite &mesh, FVArray<double> &polution) {
	//
	// STEP 1
	//

	MPI_Status status;
	// each proc sends to the left side
	if (id > 0 && mesh.left_cell_count > 0) {
		for(unsigned int i = 0; i < mesh.left_cell_count; ++i) {
			unsigned int edge = mesh.left_index_to_edge[0][i];
			unsigned int cell = mesh.edge_left_cells[edge];
			mesh.left_cells_send[0][i] = polution[cell];
		}

		// TODO send to left side
		//cout << id << " sending " << mesh.left_cell_count << " from " << id << " to " << id - 1 << endl;
		MPI_Send(&mesh.left_cells_send[0][0], mesh.left_cells_send->size(), MPI_DOUBLE, id - 1, TAG_LEFT_COMM, MPI_COMM_WORLD);
	}

	// and each proc receives from the right side
	if (id < size - 1 && mesh.right_cell_count > 0) {
		//cout << id <<" receiving " << mesh.right_cell_count << " from " << id + 1 << " to " << id << endl;
		MPI_Recv(&mesh.right_cells_recv[0][0], mesh.right_cells_recv->size(), MPI_DOUBLE, id + 1, TAG_LEFT_COMM, MPI_COMM_WORLD, &status);
	}

	//
	// STEP 2
	//
	
	// each proc sends to the right side
	if (id < size - 1 && mesh.right_cell_count > 0) {
		for(unsigned int i = 0; i < mesh.right_cell_count; ++i) {
			unsigned int edge = mesh.right_index_to_edge[0][i];
			unsigned int cell = mesh.edge_left_cells[edge];
			mesh.right_cells_send[0][i] = polution[cell];
		}
		
		// TODO send to id + 1
		//cout << id << " sending " << mesh.right_cell_count << " from " << id << " to " << id + 1 << endl;
		MPI_Send(&mesh.right_cells_send[0][0], mesh.right_cells_send->size(), MPI_DOUBLE, id + 1, TAG_RIGHT_COMM, MPI_COMM_WORLD);
	}

	if (id > 0 && mesh.left_cell_count > 0) {
		// TODO recv from left side
		//cout << id <<  " receiving " << mesh.left_cell_count << " from " << id - 1 << " to " << id << endl;
		MPI_Recv(&mesh.left_cells_recv[0][0], mesh.left_cells_recv->size(), MPI_DOUBLE, id - 1, TAG_RIGHT_COMM, MPI_COMM_WORLD, &status);
	}
}

/* compute flux kernel */
void compute_flux(FVMesh2D_SOA_Lite &mesh, FVArray<double> &flux, double dc) {
	for(unsigned edge = 0; edge < mesh.num_edges; ++edge) {
		double polu;
		double v = mesh.edge_velocity[edge];

		//polu_left	= polution[ mesh->edge_left_cells[edge] ];
		//polu_right	= (mesh->edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh->edge_right_cells[edge] ];

		if (v >= 0) {
			polu = mesh.polution[ mesh.edge_left_cells[edge] ];
		} else {
			switch (mesh.edge_part[edge]) {
				case  0: polu = (mesh.edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : mesh.polution[ mesh.edge_right_cells[edge] ]; break;
				case -1: polu = mesh.left_cells_recv[0][ mesh.edge_part_index[edge] ]; break;
				case  1: polu = mesh.right_cells_recv[0][ mesh.edge_part_index[edge] ]; break;
			}
		}

		flux[edge] = v * polu;
	}
}

/* update kernel */
void update(FVMesh2D_SOA_Lite &mesh, FVArray<double> &flux, double dt) {

	//cout << endl;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		unsigned int edge_limit = mesh.cell_edges_count[cell];
		for(unsigned int edge_i = 0; edge_i < edge_limit; ++edge_i) {
			unsigned int edge = mesh.cell_edges.elem(edge_i, 0, cell);

			double var = dt * flux[edge] * mesh.edge_lengths[edge] / mesh.cell_areas[cell];

			if (mesh.edge_left_cells[edge] == cell) {
				mesh.polution[cell] -= var;
			} else {
				mesh.polution[cell] += var;
			}
		}
	}
}
