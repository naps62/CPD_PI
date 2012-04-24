/**
 * \file FVMesh2D_SOA_Lite.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVMESH2D_SOA_LITE
#define _H_FVMESH2D_SOA_LITE

#include <string>
#include <vector>
using namespace std;

#include "FVL/FVGlobal.h"
#include "FVMesh2D.h"
#include "FVL/FVLog.h"
#include "FVL/CFVArray.h"
#include "FVL/CFVMat.h"
#include "FVL/CFVPoints2D.h"

namespace FVL {

	/**
	 * A Light-weight SOA implementation of a 2 dimensional Mesh
	 *
	 * Stores only adjacency information, used in polu experiment
	 * \todo remove this after PI
	 * 2 Dimensional mesh representation using 	Structure of Arrays instead of Array of Structures
	 * This class is based on CFVArray instead of FVArray (which would be intuitively more adequate) for compatibility with FVMesh2D_SOA
	 */
	class FVMesh2D_SOA_Lite {
		public:

			// EDGE INFO
			unsigned int num_edges;						///< total number of edges
			CFVArray<unsigned int> edge_index;
			CFVArray<double> edge_lengths;				///< length for each edge
			CFVArray<double> edge_velocity;
			CFVArray<unsigned int> edge_left_cells;		///< left cell of each edge
			CFVArray<unsigned int> edge_right_cells;	///< right cell of each edge (NO_RIGHT_CELL indicates a border edge where no right cell exists)

			CFVArray<int> edge_part;					///< partition where the right cell is located
			CFVArray<unsigned int> edge_part_index;
			CFVArray<unsigned int>* left_index_to_edge;	///< for each index, the corresponding edge
			CFVArray<unsigned int>* right_index_to_edge;
			unsigned int left_cells;					///< number of cells from left partition
			unsigned int right_cells;					///< number of cells from right partition

			// CELL INFO
			unsigned int num_cells;					///< total number of cells
			CFVArray<unsigned int> cell_index;
			CFVArray<double> cell_areas;			///< area for each cell
			CFVArray<unsigned int> cell_edges_count;///< number of edges of each cell (to index cell_edges)
			CFVMat<unsigned int> cell_edges;		///< index of edges for each cell (CFVMat(MAX_EDGES_PER_CELL, 1, num_cells)

			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/

			/**
			 * Constructor to create an empty mesh, given its dimensions
			 *
			 * When using this constructor, vertex_cells is unavailable (allocated as size 0 array)
			 *
			 * \param num_edges Number of edges to alloc
			 * \param num_cells Number of cells to alloc
			 */
			FVMesh2D_SOA_Lite(unsigned int num_edges, unsigned int num_cells);

		private:

			/************************************************
			 * MEMORY MANAGEMENT
			 ***********************************************/

			/**
			 * Allocates CPU memory for data
			 */
			void alloc();
	};

}

#endif // define _H_FVMESH2D_SOA_LITE



