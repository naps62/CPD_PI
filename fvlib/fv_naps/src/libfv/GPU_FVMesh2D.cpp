#include "GPU_FVMesh2D.h"
#include "FVLib_config.h"

#include "MFVErr.h"

/************************************************
 * CONSTRUCTORS
 ***********************************************/

GPU_FVMesh2D::GPU_FVMesh2D() {
	logger << "GPU_FVMesh2D()" << endl;
	init();
}

GPU_FVMesh2D::GPU_FVMesh2D(FVMesh2D &msh) {
	logger << "GPU_FVMesh2D(FVMesh2D &)" << endl;
	import_FVMesh2D(msh);
}

GPU_FVMesh2D::~GPU_FVMesh2D() {
	logger << "~GPU_FVMesh2D" << endl;
	delete_cpu();
}

void GPU_FVMesh2D::init() {
	logger << "GPU_FVMesh::init()" << endl;
	num_cells = 0;
	num_edges = 0;
}

/************************************************
 * IMPORT/EXPORT METHODS
 ***********************************************/

void GPU_FVMesh2D::import_FVMesh2D(FVMesh2D &msh) {
	logger << "importing FVMesh2D" << endl;
	num_edges = msh.getNbEdge();
	num_cells = msh.getNbCell();

	// allocs space for all needed data
	alloc_cpu();

	// copy edge data
	FVEdge2D *edge;
	int i = 0;
	for(msh.beginEdge(); (edge = msh.nextEdge()); ++i) {
		// edge normal
		edge_normals.x.cpu_ptr[i]	= edge->normal.x;
		edge_normals.y.cpu_ptr[i]	= edge->normal.y;

		// edge length
		edge_lengths.cpu_ptr[i] 	= edge->length;

		// edge left cell (always exists)
		edge_left_cells.cpu_ptr[i]	= edge->leftCell->label - 1;

		// edge right cell (need check. if border edge, rightCell is null)
		edge_right_cells.cpu_ptr[i]	= (edge->rightCell != NULL) ? (edge->rightCell->label - 1) : NO_RIGHT_EDGE; 
	}

	// copy cell data
	FVCell2D *cell;
	i = 0;
	for(msh.beginCell(); (cell = msh.nextCell()); ++i) {
		// cell area
		cell_areas.cpu_ptr[i]	= cell->area;
	}
}


/************************************************
 * MEMORY MANAGEMENT METHODS
 ***********************************************/
void GPU_FVMesh2D::alloc_cpu() {
	if (num_edges <= 0 || num_cells <= 0) {
		string msg = "num edges/cells not valid for allocation";
		FVErr::error(msg, -1);
	}

	logger << "allocating cpu ptrs" << endl;

	// alloc edge info
	edge_normals.alloc_cpu(num_edges);
	edge_lengths.alloc_cpu(num_edges);
	edge_left_cells.alloc_cpu(num_edges);
	edge_right_cells.alloc_cpu(num_edges);

	// alloc cell info
	cell_areas.alloc_cpu(num_cells);
}

void GPU_FVMesh2D::delete_cpu() {
	logger << "deleting cpu ptrs" << endl;

	// delete edge data
	edge_normals.delete_cpu();
	edge_lengths.delete_cpu();
	edge_left_cells.delete_cpu();
	edge_right_cells.delete_cpu();

	// delete cell data
	cell_areas.delete_cpu();
}
