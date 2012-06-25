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
			CFVArray<unsigned int> edge_index;			///< for each edge in this partition, gives it's index in the global mesh
			CFVArray<double> edge_lengths;				///< length for each edge
			CFVArray<double> edge_velocity;				///< edge velocities. moved here to better handle partition generation (velocity is also partitioned)
			CFVArray<unsigned int> edge_left_cells;		///< left cell of each edge
			CFVArray<unsigned int> edge_right_cells;	///< right cell of each edge (NO_RIGHT_CELL indicates a border edge where no right cell exists)

			CFVArray<int> edge_part;					///< for each edge, 0 if it's right cell is in this partition, -1 if it's to the left, 1 if it's to the right
			CFVArray<unsigned int> edge_part_index;		///< for each edge with edge_part == -1 or 1, gives it's the index on the communication array (where neighbor polution is read from)
			CFVArray<unsigned int>* left_index_to_edge;	///< for each index in the communication array from the left partition, the corresponding edge
			CFVArray<unsigned int>* right_index_to_edge;///< same thing, but for right side communication array
			unsigned int left_cell_count;				///< number of cells from left partition
			unsigned int right_cell_count;				///< number of cells from right partition
			CFVArray<double>* left_cells_send;	///< array to send data to left partition
			CFVArray<double>* right_cells_send;	///< array to send data to right partition
			CFVArray<double>* left_cells_recv;	///< array to receive data from left partition
			CFVArray<double>* right_cells_recv;	///< array to receive data from right partition

			// CELL INFO
			unsigned int num_cells;					///< total number of cells
			CFVArray<unsigned int> cell_index;		///< for each cell, gives it's index in the global mesh
			CFVArray<double> cell_areas;			///< area for each cell
			CFVArray<unsigned int> cell_edges_count;///< number of edges of each cell (to index cell_edges)
			CFVMat<unsigned int> cell_edges;		///< index of edges for each cell (CFVMat(MAX_EDGES_PER_CELL, 1, num_cells)
			CFVArray<double> polution;

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



