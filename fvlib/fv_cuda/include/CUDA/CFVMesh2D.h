/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** CFVMesh2D.h
** CUDA 2D Mesh
**
** Author: Miguel Palhas, mpalhas@gmail.com
** -------------------------------------------------------------------------*/

#ifndef _H_CUDA_FVMESH2D
#define _H_CUDA_FVMESH2D

#include <string>
#include <vector>
#include "FVVertex2D.h"
#include "FVCell2D.h"
#include "FVEdge2D.h"

#include "FVLib_config.h"
#include "FVMesh2D.h"
#include "XML.h"
#include "MFVLog.h"
#include "CFVVect.h"
#include "CFVPoints2D.h"

using namespace std;
class Gmsh;

namespace CudaFV {
	class CFVMesh2D {
		public:
			/**
			 * EDGE INFO
			 */
			unsigned int num_edges;
			CFVPoints2D edge_normals;				// normals for each edge
			CFVVect<double> edge_lengths;			// length for each edge
			CFVVect<unsigned int> edge_left_cells;	// size = num_edges. holds index of the cell left of each edge
			CFVVect<unsigned int> edge_right_cells;	// size = num_edges. holds index of the cell right of each edge

			/**
			 * CELL INFO
			 */
			unsigned int num_cells;
			CFVVect<double> cell_areas;	// area of each cell

		public:

			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			CFVMesh2D();
			CFVMesh2D(FVMesh2D &msh);
			~CFVMesh2D();

		private:

			/************************************************
			 * IMPORT/EXPORT
			 ***********************************************/

			/**
			 * imports a default FVMesh2D format to GPU format 
			 */
			void import_FVMesh2D(FVMesh2D &);


			/************************************************
			 * MEMORY MANAGEMENT
			 ***********************************************/

			/**
			 * Allocates CPU memory for data
			 */
			void alloc();

			/**
			 * deallocates all cpu memory
			 */
			void dealloc();

			/* Not yet implemented */
			//void complete_data();
	};

}

#endif // define _H_CUDA_FVMESH2D



