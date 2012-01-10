#include "GPU_FVMesh2D"

#include "FVErr.h"

/************************************************
 * CONSTRUCTORS
 ***********************************************/

GPU_FVMesh2D::GPU_FVMesh2D() {
	logger << "GPU_FVMesh2D()" << endl;
	num_cells = 0;
	num_edges = 0;

	zero_ptrs();
}

GPU_FVMesh2D::GPU_FVMesh2D(FVMesh2D &msh) {
	logger << "GPU_FVMesh2D(FVMesh2D &)" << endl;
	importFVMesh2D(msh);
}

GPU_FVMesh2D::~GPU_FVMesh2D() {
	logger << "~GPU_FVMesh2D" << endl;
	delete_cpu();
}

/************************************************
 * IMPORT/EXPORT METHODS
 ***********************************************/

GPU_FVMesh2D::importFVMesh2D(const FVMesh2D &msh) {
	logger << "importing FVMesh2D" << endl;
	num_edges = msg.getNbEdge();
	num_cells = msh.getNbCell();

	// allocs space for all needed data
	alloc_cpu(num_edges, num_cells);

	// copy edge data
	for(FVEdge2D *edge, int i=0, msh.beginEdge(); edge = msh.nextEdge(); ++i) {
		// edge normal
		edge_normals.x.cpu_ptr[i]	= edge->normal.x;
		edge_normals.y.cpu_ptr[i]	= edge->normal.y;

		// edge length
		edge_lengths.cpu_ptr[i] 	= edge->length;

		// edge left and right cells
		edge_left_cells.cpu_ptr[i]	= edge->leftCell->label - 1;
		edge_right_cells.cpu_ptr[i]	= edge->rightCell->label - 1;
	}
	
	// copy cell data
	for(FVCell2D *cell, int i=0, msh.beginCell(); cell = msh.nextCell(); ++i) {
		// cell area
		cel_areas.cpu_ptr[i]	= cell->area
	}
}


/************************************************
 * MEMORY MANAGEMENT METHODS
 ***********************************************/

GPU_FVMesh2D::zero_ptrs() {
	logger << "zeroing ptrs" << endl;

	// zero edge data
	zero_dualPtr(edge_normals.x);
	zero_dualPtr(edge_normals.y);
	zero_dualPtr(edge_lengths);
	zero_dualPtr(edge_left_cells);
	zero_dualPtr(edge_right_cells);

	// zero cell data
	zero_dualPtr(cell_areas);
}

GPU_FVMesh2D::zero_dualPtr(DualPtr &ptr) {
	ptr.cpu_ptr = NULL;
	ptr.gpu_ptr = NULL;
}

GPU_FVMesh2D::alloc_cpu() {
	if (num_edges <= 0 || num_cells <= 0) {
		FVerr.error("num edges/cells not valid for allocation");
	}

	logger << "allocating cpu ptrs" << endl;

	// alloc edge info
	alloc_cpu(edge_normals.x,	num_edges);
	alloc_cpu(edge_normals.x,	num_edges);
	alloc_cpu(edge_lengths,		num_edges);
	alloc_cpu(edge_left_cells,	num_edges);
	alloc_cpu(edge_right_cells,	num_edges);

	// alloc cell info
	alloc_cpu(cell_areas,	 	num_cells);
}

GPU_FVMesh2D::alloc_cpu(DualPtr &ptr, unsigned int size) {
	ptr.cpu_ptr = new double[size];
}

GPU_FVMesh2D::delete_cpu() {
	logger << "deleting cpu ptrs" << endl;

	// delete edge data
	delete_cpu(edge_normals.x);
	delete_cpu(edge_normals.y);
	delete_cpu(edge_lengths);
	delete_cpu(edge_left_cells);
	delete_cpu(edge_right_cells);

	// delete cell data
	delete_cpu(cell_areas);
}

GPU_FVMesh2D::delete_cpu(DualPtr &ptr) {
	if (ptr.cpu != NULL)
		delete ptr.cpu_ptr;
}

