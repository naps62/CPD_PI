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

#include "FVL/FVGlobal.h"
#include "FVMesh2D.h"
#include "FVL/FVLog.h"
#include "FVL/CFVVect.h"
#include "FVL/CFVPoints2D.h"

using namespace std;
class Gmsh;

namespace FVL {
	class CFVMesh2D {
		public:
			/**
			 * VERTEX INFO
			 */
			unsigned int num_vertex;
			CFVPoints2D vertex_coords;				// coords for each vertex

			/**
			 * EDGE INFO
			 */
			unsigned int num_edges;
			CFVPoints2D edge_normals;				// normals for each edge
			CFVPoints2D edge_centroids;				// centroid
			CFVVect<double> edge_lengths;			// length for each edge
			CFVVect<unsigned int> edge_fst_vertex;	// first vertex of each edge
			CFVVect<unsigned int> edge_snd_vertex;	// second vertex of each edge
			CFVVect<unsigned int> edge_left_cells;	// size = num_edges. holds index of the cell left of each edge
			CFVVect<unsigned int> edge_right_cells;	// size = num_edges. holds index of the cell right of each edge

			/**
			 * CELL INFO
			 */
			unsigned int num_cells;
			//unsigned int num_total_edges;
			CFVPoints2D cell_centroids;					// centroid of each cell
			CFVVect<double> cell_perimeters;			// perimeter of each cell
			CFVVect<double> cell_areas;					// area of each cell
			//CFVVect<unsigned int> cell_edges;			// list of edges for all cells
			//CFVVect<unsigned int> cell_edges_index;		// for each cell, index of it's edge list on cell_edges
			CFVVect<unsigned int> cell_edges_count;		// number of edges of each cell (to index cell_edges)
			vector<CFVVect<unsigned int> > cell_edges;	// index of edges for each cella
			vector<CFVPoints2D> cell2edges;				// distance of each cell to each edge

		public:

			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			CFVMesh2D();
			CFVMesh2D(FVMesh2D &msh);
			CFVMesh2D(const string &filename);
			~CFVMesh2D();

		private:

			/************************************************
			 * IMPORT/EXPORT
			 ***********************************************/

			/**
			 * imports a default FVMesh2D format to GPU format 
			 */
			void import_FVMesh2D(FVMesh2D &);
			void read_mesh_file(const string &filename);
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

			/* Not yet implemented */
			//void complete_data();
	};

}

#endif // define _H_CUDA_FVMESH2D



