/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** CFVMesh2D.h
** CUDA enabled 2D Mesh
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Test:	---
** -------------------------------------------------------------------------*/

#ifndef _H_CUDA_FVMESH2D
#define _H_CUDA_FVMESH2D

#include <string>
#include <vector>

#include "FVL/FVGlobal.h"
#include "FVMesh2D.h"
#include "FVL/FVLog.h"
#include "FVL/CFVVect.h"
#include "FVL/CFVMat.h"
#include "FVL/CFVPoints2D.h"

#include <cuda.h>


using namespace std;
class Gmsh;

namespace FVL {
	class CFVMesh2D {
		public:

			// TODO move this to a more suitable class
			typedef enum e_EdgeType {
				EDGE			= 0,
				EDGE_DIRICHLET	= 1,
				EDGE_NEUMMAN	= 2,
			} EdgeType;

			typedef enum e_CellType {
				CELL = 10,
			} CellType;

			// VERTEX INFO
			unsigned int num_vertex;
			CFVPoints2D<double> vertex_coords;		// coords for each vertex

			// EDGE INFO
			unsigned int num_edges;
			CFVVect<int> edge_types;				// type associated with each edge
			CFVPoints2D<double> edge_normals;		// normals for each edge
			CFVPoints2D<double> edge_centroids;		// centroid
			CFVVect<double> edge_lengths;			// length for each edge
			CFVVect<unsigned int> edge_fst_vertex;	// first vertex of each edge
			CFVVect<unsigned int> edge_snd_vertex;	// second vertex of each edge
			CFVVect<unsigned int> edge_left_cells;	// size = num_edges. holds index of the cell left of each edge
			CFVVect<unsigned int> edge_right_cells;	// size = num_edges. holds index of the cell right of each edge

			// CELL INFO
			unsigned int num_cells;
			CFVVect<int> cell_types;				// type associated with each cell
			CFVPoints2D<double> cell_centroids;		// centroid of each cell
			CFVVect<double> cell_perimeters;		// perimeter of each cell
			CFVVect<double> cell_areas;				// area of each cell
			CFVVect<unsigned int> cell_edges_count;	// number of edges of each cell (to index cell_edges)
			CFVMat<unsigned int> cell_edges;		// index of edges for each cell (CFVMat(MAX_EDGES_PER_CELL, 1, num_cells)
			CFVMat<double> cell_edges_normal;		// distance of each cell to each edge (CFVMat(MAX_EDGES_PER_CELL, 2, num_cells)

			

			typedef struct s_CFVMesh2D {
				// TODO this
				unsigned int num_vertex;
				unsigned int num_edges;
				unsigned int num_cells;

				double *vertex_coords[2];

				int *edge_types;
				double *edge_normals[2];
				double *edge_centroids[2];
				double *edge_lengths;
				unsigned int *edge_fst_vertex;
				unsigned int *edge_snd_vertex;
				unsigned int *edge_left_cells;
				unsigned int *edge_right_cells;

				int *cell_types;
				double *cell_centroids[2];
				double *cell_perimeters;
				double *cell_areas;
				unsigned int *cell_edges_count;
				unsigned int **cell_edges;
				double **cell_edges_normal;
			} CFVMesh2D_cuda;

			// cuda structure holding the mesh (pointer to CUDA memory)
			CFVMesh2D_cuda *cuda_mesh;

		public:

			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			CFVMesh2D();
			CFVMesh2D(FVMesh2D &msh);
			CFVMesh2D(const string &filename);
			~CFVMesh2D();

			/************************************************
			 * MEM MANAGEMENT
			 ***********************************************/

			// store all mesh data to cuda
			CFVMesh2D_cuda* cuda_malloc();

			// saves entire mesh to cuda memory
			void cuda_save(cudaStream_t stream = 0);

			// free all cuda storage of this mesh
			void cuda_free();
		private:

			/************************************************
			 * IMPORT/EXPORT
			 ***********************************************/

			// imports a default FVMesh2D format to GPU format
			void import_FVMesh2D(FVMesh2D &);

			// read mesh data from a file
			void read_mesh_file(const string &filename);

			// aux function to compute mesh values from input data (centroids, normals, areas, etc)
			void compute_final_data();


			/************************************************
			 * MEMORY MANAGEMENT
			 ***********************************************/

			// Allocates CPU memory for data
			void alloc();

			// deallocates all cpu memory
			void dealloc();
	};

}

#endif // define _H_CUDA_FVMESH2D



