/**
 * \file FVMesh2D_SOA.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVMESH2D_SOA
#define _H_FVMESH2D_SOA

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
	 * An SOA implementation of a 2 dimensional Mesh
	 *
	 * 2 Dimensional mesh representation using 	Structure of Arrays instead of Array of Structures
	 * This class is based on CFVArray instead of FVArray (which would be intuitively more adequate) for compatibility with FVMesh2D_SOA
	 */
	class FVMesh2D_SOA {
		public:
			// VERTEX INFO
			unsigned int num_vertex;				///< total number of vertex
			CFVPoints2D<double> vertex_coords;		///< coords for each vertex
			/// \todo finish this
			CFVArray<double> vertex_cells_count;	///< number of cells adjacent to each vertex TODO
			CFVArray<double> vertex_cells_index;	///< index of list of cells for each vertex TODO
			CFVArray<double> vertex_cells;			///< list of all adjacent cells for each vertex. Indexed by vertex_cells_count and vertex_cells_index TODO

			// EDGE INFO
			unsigned int num_edges;					///< total number of edges
			CFVArray<int> edge_types;				///< type associated with each edge
			CFVPoints2D<double> edge_normals;		///< normals for each edge
			CFVPoints2D<double> edge_centroids;		///< centroid for each edge
			CFVArray<double> edge_lengths;			///< length for each edge
			CFVArray<unsigned int> edge_fst_vertex;	///< first vertex of each edge
			CFVArray<unsigned int> edge_snd_vertex;	///< second vertex of each edge
			CFVArray<unsigned int> edge_left_cells;	///< left cell of each edge
			CFVArray<unsigned int> edge_right_cells;	///< right cell of each edge (NO_RIGHT_CELL indicates a border edge where no right cell exists)

			// CELL INFO
			unsigned int num_cells;					///< total number of cells
			CFVArray<int> cell_types;				///< type associated with each cell
			CFVPoints2D<double> cell_centroids;		///< centroid for each cell
			CFVArray<double> cell_perimeters;		///< perimeter for each cell
			CFVArray<double> cell_areas;			///< area for each cell
			CFVArray<unsigned int> cell_edges_count;///< number of edges of each cell (to index cell_edges)
			CFVMat<unsigned int> cell_edges;		///< index of edges for each cell (CFVMat(MAX_EDGES_PER_CELL, 1, num_cells)
			CFVMat<double> cell_edges_normal;		///< distance of each cell to each edge (CFVMat(MAX_EDGES_PER_CELL, 2, num_cells)


			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/

			/**
			 * Constructor to import a mesh on original FVMesh2D format
			 *
			 * \param msh Mesh to import
			 */
			FVMesh2D_SOA(FVMesh2D &msh);

			/**
			 * Constructor to import a mesh from a XML file
			 *
			 * \param filename XML file to import
			 */
			FVMesh2D_SOA(const string &filename);

		private:

			/************************************************
			 * IMPORT/EXPORT
			 ***********************************************/

			/**
			 * Imports data from #FVMesh2D format
			 *
			 * \param mesh #FVMesh2D to import
			 * \todo import cell_edges (and possibly others)
			 */
			void import_FVMesh2D(FVMesh2D &mesh);

			/**
			 * Read mesh data from a file
			 *
			 * \param filename XML file to read the mesh from
			 */
			void read_mesh_file(const string &filename);

			/**
			 * Aux function to compute mesh values from input data (centroids, normals, areas, etc)
			 */
			void compute_final_data();


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
	};

}

#endif // define _H_FVMESH2D_SOA



