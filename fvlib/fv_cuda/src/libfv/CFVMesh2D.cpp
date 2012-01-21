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
		// caso nao haja disponibilidade da sua partei = 0;
		FVCell2D *cell;
		i = 0;
		num_total_edges = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ++i) {
			// cell area
			cell_areas[i]	= cell->area;

			// index at which edges for this cell start
			cell_edges_index[i] = num_total_edges;
			// count of edges for this cell
			cell_edges_count[i] = cell->nb_edge;

			// total count of edges for cell_edges array
			num_total_edges += cell->nb_edge;
		}

		// finally create data for cell_edges array
		// this is not in alloc() func since it depends on values calculated on previous loop

		cell_edges = CFVVect<unsigned int>(num_total_edges);
		i = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ) {
			for(cell->beginEdge(); (edge = cell->nextEdge()); ++i) {
				cell_edges[i] = edge->label - 1;
			}
		}

		/*cout << "num_total_edges = "<< num_total_edges << endl;
		int j = 0;
		for(i = 0; i < num_cells; ++i) {
			cout << "cell " << i << " at " << j << " with " << cell_edges_count[i] << "edges:\t";
			for(j = 0; j < cell_edges_count[i]; ++j) {
				cout << cell_edges[ cell_edges_index[i] + j ] << "     ";
			}
			cout << endl;
		}
		exit(0);*/
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
		edge_normals		= CFVPoints2D(num_edges);
		edge_lengths		= CFVVect<double>(num_edges);
		edge_left_cells		= CFVVect<unsigned int>(num_edges);
		edge_right_cells	= CFVVect<unsigned int>(num_edges);

		// alloc cell info
		cell_areas			= CFVVect<double>(num_cells);
		cell_edges_index	= CFVVect<unsigned int>(num_cells);
		cell_edges_count	= CFVVect<unsigned int>(num_cells);
	}
}

