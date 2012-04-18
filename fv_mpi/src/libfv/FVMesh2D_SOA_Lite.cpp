#include "FVL/FVMesh2D_SOA_Lite.h"
#include "FVLib_config.h"
#include "FVPoint2D.h"

#include "FVL/FVErr.h"

using namespace rapidxml;

namespace FVL {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	FVMesh2D_SOA_Lite::FVMesh2D_SOA_Lite(unsigned int num_edges, unsigned int num_cells) {
		this->num_edges	 = num_edges;
		this->num_cells  = num_cells;

		alloc();
	}

	/************************************************
	 * MEMORY MANAGEMENT METHODS
	 ***********************************************/
	void FVMesh2D_SOA_Lite::alloc() {
		if (num_vertex <= 0 || num_edges <= 0 || num_cells <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		// alloc edge info
		edge_lengths		= CFVArray<double>(num_edges);
		edge_left_cells		= CFVArray<unsigned int>(num_edges);
		edge_right_cells	= CFVArray<unsigned int>(num_edges);

		// alloc cell info
		cell_areas			= CFVArray<double>(num_cells);
		cell_edges_count	= CFVArray<unsigned int>(num_cells);
		cell_edges			= CFVMat<unsigned int>(MAX_EDGES_PER_CELL, 1, num_cells);
	}
}

