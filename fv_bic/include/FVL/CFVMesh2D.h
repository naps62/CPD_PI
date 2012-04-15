/**
 * \file CFVMesh2D.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_CFVMESH2D
#define _H_CFVMESH2D

#ifndef __CUDACC__
#error CFVMesh2D is not supported outside of a CUDA environment
#endif

#include <string>
#include <vector>
using namespace std;

#include <cuda.h>

#include "FVL/FVGlobal.h"
#include "FVMesh2D.h"
#include "FVL/FVLog.h"
#include "FVL/CFVArray.h"
#include "FVL/CFVMat.h"
#include "FVL/CFVPoints2D.h"

namespace FVL {


	#ifdef __CUDACC__
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
	#endif // __CUDACC__


	/**
	 * A CUDA enabled 2D Mesh
	 *
	 * 2 Dimensional mesh representation using 	Structure of Arrays instead of Array of Structures
	 */
	class CFVMesh2D : public FVMesh2D_SOA {
		private:
			#ifdef __CUDACC__
			CFVMesh2D_cuda *cuda_mesh;	///< CUDA structure holding the mesh (ptr to CUDA memory space). Set to NULL if cuda memory not allocated
			#endif

		public:
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
			 * CUDA
			 ***********************************************/

			/**
			 * Returns a pointer to device memory containing the mesh in #CFVMesh2D_cuda format
			 *
			 * If device memory is already allocated, nothing is done, and the current pointer is returned unaltered. Use cuda_free() first to force mesh reallocation.
			 *
			 * \return Pointer to #CFVMesh2D_cuda struct, or NULL if no memory was previously allocated on the device
			 */
			CFVMesh2D_cuda *cuda_get();

			/**
			 * Checks whether device memory is allocated
			 *
			 * \return true if memory is allocated on the CUDA device, false otherwise
			 */
			bool cuda_is_alloc();

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
	};

}

#endif // define _H_CUDA_FVMESH2D



