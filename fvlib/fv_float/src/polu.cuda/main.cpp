#include <iostream>
#include <cmath>
#include <cfloat>
#include "FVLib.h"
#include "CUDA/CFVLib.h"

#include "parameters.h"
#include "polu_cuda.h"

/*
   Computes the mesh parameter (whatever that is)
   */
fv_float compute_mesh_parameters(FVMesh2D mesh);

/*
   Função Madre
   */
int main() {
	string name;
	fv_float h;
	fv_float t;
	FVMesh2D mesh;

	Parameters data;
	data = read_parameters( "param.xml" );

	// read the mesh
	mesh.read( data.filenames.mesh.c_str() );

	// GPU
	CudaFV::CFVMesh2D gpu_mesh(mesh);

	FVVect<fv_float> polution( mesh.getNbCell() );
	FVVect<fv_float> flux( mesh.getNbEdge() );
	FVVect<FVPoint2D<fv_float> > velocity( mesh.getNbCell() );

	//	read veloci
	FVio velocity_file( data.filenames.velocity.c_str() , FVREAD );
	velocity_file.get( velocity , t , name );

	// GPU
	CudaFV::CFVVect<fv_float> gpu_polution(gpu_mesh.num_cells);
	CudaFV::CFVVect<fv_float> gpu_flux(gpu_mesh.num_edges);
	CudaFV::CFVPoints2D gpu_velocity(gpu_mesh.num_cells);

	for(unsigned int i = 0; i < gpu_polution.size(); ++i) {
		gpu_polution[i] = polution[i];
		gpu_velocity.x[i] = velocity[i].x;
		gpu_velocity.y[i] = velocity[i].y;
	}

	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVREAD );
	polu_ini_file.get( polution , t , name );

	h = compute_mesh_parameters( mesh );

	// GPU
	cuda_main_loop(
			data.time.final,
			data.iterations.jump,
			gpu_mesh,
			h,
			polution,
			gpu_polution,
			gpu_velocity,
			gpu_flux,
			data.computation.threshold);

	return 0;
}


fv_float compute_mesh_parameters (FVMesh2D mesh) {
	fv_float h;
	fv_float S;
	FVCell2D *cell;
	FVEdge2D *edge;

	h = 1.e20;
	for ( mesh.beginCell(); ( cell = mesh.nextCell() ) ; ) {
		S = cell->area;
		for ( cell->beginEdge(); ( edge = cell->nextEdge() ) ; ) {
			if ( h * edge->length > S )
				h = S / edge->length;
		}
	}
	return h;
}
