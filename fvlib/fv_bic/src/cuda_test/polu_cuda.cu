#include <cuda.h>

#include "FVLib.h"
#include "CUDA/CFVLib.h"
#include "FVLib.h"

#include "parameters.h"
#include "polu_cuda.h"
#include "kernels.cuh"


void gpu_update(
		CudaFV::CFVMesh2D &mesh,
		CudaFV::CFVVect<double> &polution,
		CudaFV::CFVVect<double> &flux,
		double dt) {

	// PARA CADA EDGE
	// 		POLUTION DA CELL A ESQUERDA: 
	for (unsigned int i = 0; i < mesh.num_edges; ++i) {
		polution[ (unsigned int) mesh.edge_left_cells[i] ] -=
			dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_left_cells[i] ];
		if (mesh.edge_right_cells[i] != NO_RIGHT_EDGE)
			polution[ (unsigned int) mesh.edge_right_cells[i] ] +=
				dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_right_cells[i] ];
	}
}

double compute_mesh_parameters (FVMesh2D mesh) {
	double h;
	double S;
	FVCell2D *cell;
	FVEdge2D *edge;

	h = 1.e20;
	for ( mesh.beginCell(); (cell = mesh.nextCell()) != NULL ; ) {
		S = cell->area;
		for ( cell->beginEdge(); (edge = cell->nextEdge()) != NULL; ) {
			if ( h * edge->length > S )
				h = S / edge->length;
		}
	}
	return h;
}

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
int main() {

	string name;
	double t;
	FVMesh2D old_mesh;

	Parameters data;
	data = read_parameters( "param.xml" );

	// read the mesh
	old_mesh.read( data.filenames.mesh.c_str() );

	// GPU
	CudaFV::CFVMesh2D mesh(data.filenames.mesh);

	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		cout << mesh.edge_normals.x[i] << " " << mesh.edge_normals.y[i] << " " << mesh.edge_lengths[i] << endl;	
	}
	exit(0);
	//CudaFV::CFVMesh2D mesh(old_mesh);

	FVVect<double> old_polution( old_mesh.getNbCell() );
	FVVect<double> old_flux( old_mesh.getNbEdge() );
	FVVect<FVPoint2D<double> > old_velocity( old_mesh.getNbCell() );

	//	read veloci
	FVio velocity_file( data.filenames.velocity.c_str() , FVREAD );
	velocity_file.get( old_velocity , t , name );

	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVREAD );
	polu_ini_file.get( old_polution , t , name );
	
	return 0;
}

