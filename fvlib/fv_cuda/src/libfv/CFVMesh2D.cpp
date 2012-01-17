#include "CUDA/CFVMesh2D.h"
#include "FVLib_config.h"

#include "MFVErr.h"

namespace CudaFV {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	CFVMesh2D::CFVMesh2D() {
		FVLog::logger << "CFVMesh2D()" << endl;
	}

	CFVMesh2D::CFVMesh2D(FVMesh2D &msh) {
		FVLog::logger << "CFVMesh2D(FVMesh2D &)" << endl;
		import_FVMesh2D(msh);
	}

	CFVMesh2D::~CFVMesh2D() {
		FVLog::logger << "~CFVMesh2D" << endl;
	}

	/************************************************
	 * IMPORT/EXPORT METHODS
	 ***********************************************/

	void CFVMesh2D::import_FVMesh2D(FVMesh2D &msh) {
		FVLog::logger << "importing FVMesh2D" << endl;
		num_edges = msh.getNbEdge();
		num_cells = msh.getNbCell();

		// allocs space for all needed data
		alloc();

		// copy edge data
		FVEdge2D *edge;
		int i = 0;
		for(msh.beginEdge(); (edge = msh.nextEdge()); ++i) {
			// edge normal
			edge_normals.x[i]	= edge->normal.x;
			edge_normals.y[i]	= edge->normal.y;

			// edge length
			edge_lengths[i] 	= edge->length;

			// edge left cell (always exists)
			edge_left_cells[i]	= edge->leftCell->label - 1;

			// edge right cell (need check. if border edge, rightCell is null)
			edge_right_cells[i]	= (edge->rightCell != NULL) ? (edge->rightCell->label - 1) : NO_RIGHT_EDGE; 
		}

		// copy cell data
		FVCell2D *cell;
		i = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ++i) {
			// cell area
			cell_areas[i]	= cell->area;
		}
	}


	/************************************************
	 * MEMORY MANAGEMENT METHODS
	 ***********************************************/
	void CFVMesh2D::alloc() {
		if (num_edges <= 0 || num_cells <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		FVLog::logger << "allocating cpu ptrs" << endl;
		// alloc edge info
		edge_normals = CFVPoints2D(num_edges);
		edge_lengths = CFVVect<fv_float>(num_edges);
		edge_left_cells  = CFVVect<unsigned int>(num_edges);
		edge_right_cells = CFVVect<unsigned int>(num_edges);

		// alloc cell info
		cell_areas = CFVVect<fv_float>(num_cells);
	}
}
