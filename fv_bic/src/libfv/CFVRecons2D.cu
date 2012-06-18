#include <fstream>
#include <string>

#include "FVL/CFVRecons2D.h"

namespace FVL {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	CFVRecons2D::CFVRecons2D(FVMesh2D_SOA &msh) : FVRecons2D_SOA(msh) {
		cuda_recons = NULL;
	}

	CFVRecons2D::~CFVRecons2D() {
		if (cuda_is_alloc())
			cuda_free();
	}

	/************************************************
	 * CUDA
	 ***********************************************/
	CFVRecons2D_cuda* CFVRecons2D::cuda_get() {
		return cuda_recons;
	}

	bool CFVRecons2D::cuda_is_alloc() {
		return (cuda_recons != NULL);
	}

	CFVRecons2D_cuda* CFVRecons2D::cuda_malloc() {
		// if cuda memory is already allocated, skip and return it
		if (! cuda_is_alloc()) {
			CFVRecons2D_cuda tmp_cuda_recons;

			tmp_cuda_recons.u_ij = u_ij.cuda_malloc();
			tmp_cuda_recons.u_ji = u_ji.cuda_malloc();
			tmp_cuda_recons.F_ij = F_ij.cuda_malloc();
			tmp_cuda_recons.F_ij_old = F_ij_old.cuda_malloc();
			tmp_cuda_recons.cell_state = cell_state.cuda_malloc();
			tmp_cuda_recons.edge_state = edge_state.cuda_malloc();

			// CFVRecons2D_cuda allocation
			cudaMalloc(&cuda_recons, sizeof(CFVRecons2D_cuda));
			cudaMemcpy(cuda_recons, &tmp_cuda_recons, sizeof(CFVRecons2D_cuda), cudaMemcpyHostToDevice);
		}

		return cuda_recons;
	}

	void CFVRecons2D::cuda_save(cudaStream_t stream) {
		u_ij.cuda_save(stream);
		u_ji.cuda_save(stream);
		F_ij.cuda_save(stream);
		F_ij_old.cuda_save(stream);
		cell_state.cuda_save(stream);
		edge_state.cuda_save(stream);
	}

	void CFVRecons2D::cuda_load(cudaStream_t stream) {
		u_ij.cuda_load(stream);
		u_ji.cuda_load(stream);
		F_ij.cuda_load(stream);
		F_ij_old.cuda_load(stream);
		cell_state.cuda_load(stream);
		edge_state.cuda_load(stream);
	}

	void CFVRecons2D::cuda_free() {
		u_ij.cuda_free();
		u_ji.cuda_free();
		F_ij.cuda_free();
		F_ij_old.cuda_free();
		cell_state.cuda_free();
		edge_state.cuda_free();

		cudaFree(cuda_recons);
	}
}

