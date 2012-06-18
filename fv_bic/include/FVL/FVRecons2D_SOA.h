/**
 * \file FVRecons2D_SOA.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVRECONDS2D_SOA
#define _H_FVRECONDS2D_SOA

#include <string>
#include <vector>
using namespace std;

#include "FVL/FVGlobal.h"
#include "FVL/FVMesh2D_SOA.h"
#include "FVL/CFVArray.h"
#include "FVL/FVLog.h"
#include "FVL/FVArray.h"

namespace FVL {

	/**
	 * An SOA implementation of a 2 dimensional Mesh
	 *
	 * 2 Dimensional mesh representation using 	Structure of Arrays instead of Array of Structures
	 * This class is based on CFVArray instead of FVArray (which would be intuitively more adequate) for compatibility with FVRecons2D_SOA
	 */
	class FVRecons2D_SOA {
		public:
			//FVMesh2D_SOA   mesh;		///< the given mesh

			CFVArray<double> u_ij;		///< edge coeficient from i to j
			CFVArray<double> u_ji;		///< edge coeficient from j to i
			CFVArray<double> F_ij;		///< computed flux between i and j
			CFVArray<double> F_ij_old;	///< previous computed flux
			CFVArray<bool> cell_state;	///< current state of each cell
			CFVArray<bool> edge_state;	///< current state of each edge


			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/

			/**
			 * Constructor to create an empty mesh, given its dimensions
			 *
			 * When using this constructor, vertex_cells is unavailable (allocated as size 0 array)
			 *
			 * \param mesh Mesh to use as initializer
			 */
			FVRecons2D_SOA(FVMesh2D_SOA &mesh);

		private:

			/************************************************
			 * MEMORY MANAGEMENT
			 ***********************************************/

			/**
			 * Allocates CPU memory for data
			 */
			void alloc(unsigned int cells, unsigned int edges);
	};

}

#endif // define _H_FVRECONS2D_SOA



