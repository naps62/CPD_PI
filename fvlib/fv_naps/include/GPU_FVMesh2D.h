// ------ GPU_FVMesh2D.h ------
// S. CLAIN 2011/07
#ifndef _GPU_FVMESH2D
#define _GPU_FVMESH2D

#include <string>
#include <vector>
#include "FVVertex2D.h"
#include "FVCell2D.h"
#include "FVEdge2D.h"

#include "FVLib_config.h"
#include "FVMesh2D.h"
#include "XML.h"
#include "MFVLog.h"
#include "MFVTypes.h"

using namespace std;
class Gmsh;

class GPU_FVMesh2D {
	private:
		FVLog logger;

	public:
		/**
		 * EDGE INFO
		 */
		unsigned int num_edges;			// total number of edges
		FV_GPU_Point2D edge_normals;	// normals for each edge
		FV_DualPtr edge_lengths;		// length for each edge
		FV_DualPtr edge_left_cells;		// size = num_edges. holds index of the cell left of each edge
		FV_DualPtr edge_right_cells;	// size = num_edges. holds index of the cell right of each edge

		/**
		 * CELL INFO
		 */
		unsigned int num_cells;		// total number of cells
		FV_DualPtr cell_areas;		// area of each cell

	public:

		/************************************************
		 * CONSTRUCTORS
		 ***********************************************/
		GPU_FVMesh2D();
		GPU_FVMesh2D(FVMesh2D &msh);
		~GPU_FVMesh2D();

	private:

		/**
		 * used by constructors
		 */
		void init();
		
		/************************************************
		 * IMPORT/EXPORT METHODS
		 ***********************************************/

		/**
		 * imports a default FVMesh2D format to GPU format 
		 */
		void import_FVMesh2D(FVMesh2D &);


		/************************************************
		 * MEMORY MANAGEMENT METHODS
		 ***********************************************/

		/**
		 * Allocates CPU memory for data
		 */
		void alloc_cpu();

		/**
		 * deallocates all cpu memory
		 */
		void delete_cpu();

		/* Not yet implemented */
		//void complete_data();
};






#endif // define _GPU_FVMESH2D


