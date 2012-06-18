/**
 * \file CFVRecons2D.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_CFVRECONS2D
#define _H_CFVRECONS2D

#include "FVL/FVMesh2D_SOA.h"
#ifndef __CUDACC__
//#define CFVRecons2D FVMesh2D_SOA
#error CFVRecons2D is not available outside of a CUDA environment
#endif

#include <string>
#include <vector>
using namespace std;

#include <cuda.h>

#include "FVL/FVGlobal.h"
#include "FVRecons2D_SOA.h"
#include "FVL/FVLog.h"
#include "FVL/CFVArray.h"

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
	struct CFVRecons2D_cuda {
		CFVRecons2D_cuda *mesh;

		double *u_ij;
		double *u_ji;
		double *F_ij;
		double *F_ij_old;
		bool *cell_state;
		bool *edge_state;
	};

	/**
	 * A CUDA enabled 2D Mesh
	 *
	 * 2 Dimensional mesh representation using 	Structure of Arrays instead of Array of Structures
	 */
	class CFVRecons2D : public FVRecons2D_SOA {
		private:
			#ifdef __CUDACC__
			CFVRecons2D_cuda *cuda_recons;	///< CUDA structure holding the mesh (ptr to CUDA memory space). Set to NULL if cuda memory not allocated
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
			CFVRecons2D(FVMesh2D_SOA &msh);

			/**
			 * Constructor to import a mesh from a XML file
			 *
			 * \param filename XML file to import
			 */
			CFVRecons2D(const string &filename);

			/**
			 * Default destructor
			 *
			 * Releases all memory allocated by the mesh, both on host and device
			 */
			~CFVRecons2D();

			/************************************************
			 * CUDA
			 ***********************************************/

			/**
			 * Returns a pointer to device memory containing the mesh in #CFVRecons2D_cuda format
			 *
			 * If device memory is already allocated, nothing is done, and the current pointer is returned unaltered. Use cuda_free() first to force mesh reallocation.
			 *
			 * \return Pointer to #CFVRecons2D_cuda struct, or NULL if no memory was previously allocated on the device
			 */
			CFVRecons2D_cuda *cuda_get();

			/**
			 * Checks whether device memory is allocated
			 *
			 * \return true if memory is allocated on the CUDA device, false otherwise
			 */
			bool cuda_is_alloc();

			/**
			 * Allocate space on device memory for the entire recons
			 *
			 * \return Ptr to #CFVRecons2D_cuda structure where all data is stored
			 */
			CFVRecons2D_cuda* cuda_malloc();

			/**
			 * Saves entire recons to cuda memory
			 *
			 * Memory must have been previously allocated with cuda_malloc()
			 *
			 * \param stream CUDA Stream to use (defaults to 0 to use no stream)
			 */
			void cuda_save(cudaStream_t stream = 0);

			/**
			 * Loads recons data from cuda memory
			 *
			 * \param stream CUDA Stream to use
			 */
			void cuda_load(cudaStream_t stream = 0);

			/**
			 * Free all cuda storage of this mesh
			 */
			void cuda_free();
	};

}

#endif // define _H_CUDA_FVRECONS2D



