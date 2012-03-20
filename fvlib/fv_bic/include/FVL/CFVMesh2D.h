/**
 * \file CFVMesh2D.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_CUDA_FVMESH2D
#define _H_CUDA_FVMESH2D

#include <string>
#include <vector>
#include <cuda.h>
using namespace std;

#include "FVL/FVGlobal.h"
#include "FVMesh2D.h"
#include "FVL/FVLog.h"
#include "FVL/CFVArray.h"
#include "FVL/CFVMat.h"
#include "FVL/CFVPoints2D.h"

namespace FVL {


	/**
	 * 2D Mesh structure to use in a CUDA device
	 *
	 * In a CUDA environment (i.e. when programming a CUDA kernel) a different memory space is used (device memory) instead of RAM.
	 * There is also no access to class methods, so the following structure must be used instead to access the mesh
	 * Before usage, cuda_malloc() must be used to ensure the memory is allocated
	 * cuda_save() is also probably necessary to copy mesh data to the device
	 *
	 * \todo move this to a more suitable location
	 */
	struct CFVMesh2D_cuda {
		unsigned int num_vertex;		///< total number of vertex
		unsigned int num_edges;			///< total number of edges
		unsigned int num_cells;			///< total number of cells

		double *vertex_coords[2];		///< coords for each vertex

		int *edge_types;				///< type associated with each edge
		double *edge_normals[2];		///< normals for each edge
		double *edge_centroids[2];		///< centroid for each edge
		double *edge_lengths;			///< length for each edge
		unsigned int *edge_fst_vertex;	///< first vertex of each edge
		unsigned int *edge_snd_vertex;	///< second vertex of each edge
		unsigned int *edge_left_cells;	///< left cell of each edge
		unsigned int *edge_right_cells;	///< right cell of each edge (NO_RIGHT_CELL indicates a border edge where no right cell exists)

		int *cell_types;				///< type associated with each cell
		double *cell_centroids[2];		///< centroid for each cell
		double *cell_perimeters;		///< perimeter for each cell	
		double *cell_areas;				///< area for each cell
		unsigned int *cell_edges_count;	///< number of edges for each cell (to index cell_edges)
		unsigned int **cell_edges;		///< index of edges for each cell (unsigned int [MAX_EDGES_PER_CELL][num_cells])
		double **cell_edges_normal;		///< distance of each cell to each edge (double [2*MAX_EDGES_PER_CELL][num_cells])
	};


	/**
	 * A CUDA enabled 2D Mesh
	 *
	 * 2 Dimensional mesh representation using 	Structure of Arrays instead of Array of Structures
	 */
	class CFVMesh2D {
		private:
			CFVMesh2D_cuda *cuda_mesh;	///< CUDA structure holding the mesh (ptr to CUDA memory space). Set to NULL if cuda memory not allocated

		public:

			// VERTEX INFO
			unsigned int num_vertex;				///< total number of vertex
			CFVPoints2D<double> vertex_coords;		///< coords for each vertex

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
			CFVArray<double> cell_areas;				///< area for each cell
			CFVArray<unsigned int> cell_edges_count;	///< number of edges of each cell (to index cell_edges)
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
			CFVMesh2D(FVMesh2D &msh);

			/**
			 * Constructor to import a mesh from a XML file
			 *
			 * \param filename XML file to import
			 */
			CFVMesh2D(const string &filename);

			/**
			 * Default destructor
			 *
			 * Releases all memory allocated by the mesh, both on host and device
			 */
			~CFVMesh2D();

			/************************************************
			 * GETTERS/SETTERS
			 ***********************************************/

			/**
			 * Returns a pointer to device memory containing the mesh in #CFVMesh2D_cuda format
			 *
			 * If device memory is already allocated, nothing is done, and the current pointer is returned unaltered. Use cuda_free() first to force mesh reallocation.
			 *
			 * \return Pointer to #CFVMesh2D_cuda struct, or NULL if no memory was previously allocated on the device
			 */
			CFVMesh2D_cuda *cuda_getMesh();

			/**
			 * Checks whether device memory is allocated
			 *
			 * \return true if memory is allocated on the CUDA device, false otherwise
			 */
			bool cuda_is_alloc();


			/************************************************
			 * MEM MANAGEMENT
			 ***********************************************/

			/**
			 * Allocate space on device memory for the entire mesh
			 *
			 * \return Ptr to #CFVMesh2D_cuda structure where all data is stored
			 */
			CFVMesh2D_cuda* cuda_malloc();

			/**
			 * Saves entire mesh to cuda memory
			 *
			 * Memory must have been previously allocated with cuda_malloc()
			 *
			 * \param stream CUDA Stream to use (defaults to 0 to use no stream)
			 */
			void cuda_save(cudaStream_t stream = 0);

			/**
			 * Free all cuda storage of this mesh
			 */
			void cuda_free();
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

#endif // define _H_CUDA_FVMESH2D



