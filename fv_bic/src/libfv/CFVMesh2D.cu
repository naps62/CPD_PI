#include <fstream>
#include <string>

#include "FVL/CFVMesh2D.h"
#include "FVLib_config.h"
#include "FVPoint2D.h"

#include "rapidxml/rapidxml.hpp"
#include "FVL/FVXMLReader.h"
#include "FVL/FVErr.h"

using namespace rapidxml;

namespace FVL {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	CFVMesh2D::CFVMesh2D(FVMesh2D &msh) : FVMesh2D_SOA(msh) {
		cuda_mesh = NULL;
	}

	CFVMesh2D::CFVMesh2D(const string &filename) : FVMesh2D_SOA(filename) {
		cuda_mesh = NULL;
	}

	CFVMesh2D::~CFVMesh2D() {
		cuda_free();
	}

	/************************************************
	 * CUDA
	 ***********************************************/
	CFVMesh2D_cuda* CFVMesh2D::cuda_get() {
		return cuda_mesh;
	}

	bool CFVMesh2D::cuda_is_alloc() {
		return (cuda_mesh != NULL);
	}

	CFVMesh2D_cuda* CFVMesh2D::cuda_malloc() {
		// if cuda memory is already allocated, skip and return it
		if (! cuda_is_alloc()) {
			CFVMesh2D_cuda tmp_cuda_mesh;

			// vertex info
			tmp_cuda_mesh.num_vertex = num_vertex;
			tmp_cuda_mesh.num_edges = num_edges;
			tmp_cuda_mesh.num_cells = num_cells;

			tmp_cuda_mesh.vertex_coords[0] = vertex_coords.x.cuda_malloc();
			tmp_cuda_mesh.vertex_coords[1] = vertex_coords.y.cuda_malloc();

			// edge info
			tmp_cuda_mesh.edge_types		= edge_types.cuda_malloc();
			tmp_cuda_mesh.edge_normals[0]	= edge_normals.x.cuda_malloc();
			tmp_cuda_mesh.edge_normals[1]	= edge_normals.y.cuda_malloc();
			tmp_cuda_mesh.edge_centroids[0]	= edge_centroids.x.cuda_malloc();
			tmp_cuda_mesh.edge_centroids[1] = edge_centroids.y.cuda_malloc();
			tmp_cuda_mesh.edge_lengths		= edge_lengths.cuda_malloc();
			tmp_cuda_mesh.edge_fst_vertex	= edge_fst_vertex.cuda_malloc();
			tmp_cuda_mesh.edge_snd_vertex	= edge_snd_vertex.cuda_malloc();
			tmp_cuda_mesh.edge_left_cells	= edge_left_cells.cuda_malloc();
			tmp_cuda_mesh.edge_right_cells	= edge_right_cells.cuda_malloc();

			// cell info
			tmp_cuda_mesh.cell_types		= cell_types.cuda_malloc();
			tmp_cuda_mesh.cell_centroids[0]	= cell_centroids.x.cuda_malloc();
			tmp_cuda_mesh.cell_centroids[1]	= cell_centroids.y.cuda_malloc();
			tmp_cuda_mesh.cell_perimeters	= cell_perimeters.cuda_malloc();
			tmp_cuda_mesh.cell_areas		= cell_areas.cuda_malloc();
			tmp_cuda_mesh.cell_edges_count	= cell_edges_count.cuda_malloc();
			tmp_cuda_mesh.cell_edges		= cell_edges.cuda_malloc();
			tmp_cuda_mesh.cell_edges_normal	= cell_edges_normal.cuda_malloc();

			// CFVMesh2D_cuda allocation
			cudaMalloc(&cuda_mesh, sizeof(CFVMesh2D_cuda));
			cudaMemcpy(cuda_mesh, &tmp_cuda_mesh, sizeof(CFVMesh2D_cuda), cudaMemcpyHostToDevice);
		}

		return cuda_mesh;
	}

	void CFVMesh2D::cuda_save(cudaStream_t stream) {
		vertex_coords.x.cuda_save(stream);
		vertex_coords.y.cuda_save(stream);
		edge_normals.x.cuda_save(stream);
		edge_normals.y.cuda_save(stream);
		edge_centroids.x.cuda_save(stream);
		edge_centroids.y.cuda_save(stream);
		edge_lengths.cuda_save(stream);
		edge_fst_vertex.cuda_save(stream);
		edge_snd_vertex.cuda_save(stream);
		edge_left_cells.cuda_save(stream);
		edge_right_cells.cuda_save(stream);
		cell_centroids.x.cuda_save(stream);
		cell_centroids.y.cuda_save(stream);
		cell_perimeters.cuda_save(stream);
		cell_areas.cuda_save(stream);
		cell_edges_count.cuda_save(stream);
		cell_edges.cuda_save(stream);
		cell_edges_normal.cuda_save(stream);
	}

	void CFVMesh2D::cuda_free() {
		// vertex info
		vertex_coords.x.cuda_free();
		vertex_coords.y.cuda_free();

		// edge info
		edge_types.cuda_free();
		edge_normals.x.cuda_free();
		edge_normals.y.cuda_free();
		edge_centroids.x.cuda_free();
		edge_centroids.y.cuda_free();
		edge_lengths.cuda_free();
		edge_fst_vertex.cuda_free();
		edge_snd_vertex.cuda_free();
		edge_left_cells.cuda_free();
		edge_right_cells.cuda_free();

		// cell info
		cell_types.cuda_free();
		cell_centroids.x.cuda_free();
		cell_centroids.y.cuda_free();
		cell_perimeters.cuda_free();
		cell_areas.cuda_free();
		cell_edges_count.cuda_free();
		cell_edges.cuda_free();
		cell_edges_normal.cuda_free();

		cudaFree(cuda_mesh);
	}
}

