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

	CFVMesh2D::CFVMesh2D(FVMesh2D &msh) {
		cuda_mesh = NULL;
		import_FVMesh2D(msh);
	}

	CFVMesh2D::CFVMesh2D(const string &filename) {
		cuda_mesh = NULL;
		read_mesh_file(filename);
	}

	CFVMesh2D::~CFVMesh2D() {
		cuda_free();
	}

	/************************************************
	 * IMPORT/EXPORT METHODS
	 ***********************************************/

	void CFVMesh2D::import_FVMesh2D(FVMesh2D &msh) {
		num_edges = msh.getNbEdge();
		num_cells = msh.getNbCell();

		// allocs space for all needed data
		alloc();


		// copy edge data
		FVEdge2D *edge;
		int i = 0;
		for(msh.beginEdge(); (edge = msh.nextEdge()); ++i) {
			// edge normal
			edge_normals.x[i]	= edge->normal.x;
			edge_normals.y[i]	= edge->normal.y;

			// edge length
			edge_lengths[i] 	= edge->length;

			// edge left cell (always exists)
			edge_left_cells[i]	= edge->leftCell->label - 1;

			// edge right cell (need check. if border edge, rightCell is null)
			edge_right_cells[i]	= (edge->rightCell != NULL) ? (edge->rightCell->label - 1) : NO_RIGHT_CELL; 
		}

		// copy cell data
		FVCell2D *cell;
		i = 0;
		//num_total_edges = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ++i) {
			// cell area
			cell_areas[i]	= cell->area;

			// count of edges for this cell
			cell_edges_count[i] = cell->nb_edge;
		}
	}

	void CFVMesh2D::read_mesh_file(const string &filename) {
		FVXMLReader mesh(filename);

		// get reference for each element list
		xml_node<> *vertex	= mesh.first_node()->first_node()->first_node();
		xml_node<> *edge	= vertex->next_sibling();
		xml_node<> *cell	= edge->next_sibling();

		// get count of each element
		FVXMLReader::str_cast<unsigned int>(num_vertex, vertex->first_attribute("nbvertex")->value());
		FVXMLReader::str_cast<unsigned int>(num_edges, edge->first_attribute("nbedge")->value());
		FVXMLReader::str_cast<unsigned int>(num_cells, cell->first_attribute("nbcell")->value());

		// alloc cpu memory for all data
		alloc();

		// read vertex data
		stringstream vertex_ss(vertex->value());
		for(unsigned int i = 0; i < num_vertex; ++i) {
			int id, type;

			vertex_ss >> id;					// dummy, unused
			vertex_ss >> type;					// dummy, unused
			vertex_ss >> vertex_coords.x[i];	// x coord
			vertex_ss >> vertex_coords.y[i];	// y coord
		}
		
		// read edge data
		stringstream edge_ss(edge->value());
		for(unsigned int i = 0; i < num_edges; ++i) {
			int id, type;
			unsigned int cell_count, vertex1, vertex2;
			
			edge_ss >> id;			// dummy, unused
			edge_ss >> type;		// edge type
			edge_ss >> cell_count;	// dummy, unused
			edge_ss >> vertex1;		// x coord
			edge_ss >> vertex2;		// y coord

			edge_types[i]		= type;
			edge_fst_vertex[i]	= vertex1 - 1;
			edge_snd_vertex[i]	= vertex2 - 1;
		}

		// read cell data
		stringstream cell_ss(cell->value());
		for(unsigned int i = 0; i < num_cells; ++i) {
			int id;

			cell_ss >> id;					// read id
			id--;							// change id to 0-based index
			cell_ss >> cell_types[i];		// cell type
			cell_ss >> cell_edges_count[i];	// number of edges on this cell

			if (cell_edges_count[i] > MAX_EDGES_PER_CELL) {
				string msg("edges per cell exceed MAX_EDGES_PER_CELL. please update flag and recompile");
				FVErr::error(msg, 1);
			}

			for(unsigned int e = 0; e < cell_edges_count[i]; ++e) {
				cell_ss >> cell_edges.elem(e, 0, i);	// reads e'th edge of cell i
				cell_edges.elem(e, 0, i)--;				// change to 0-based index
			}
		}

		this->compute_final_data();
	}

	void CFVMesh2D::compute_final_data() {
		// initialize vertex
		// compute centroid and length
		for(unsigned int i = 0; i < num_edges; ++i) {
			edge_left_cells[i] = NO_CELL;
			edge_right_cells[i] = NO_CELL;

			unsigned int v1_i = edge_fst_vertex[i];
			unsigned int v2_i = edge_snd_vertex[i];

			// get vertex
			FVPoint2D<double> v1, v2;
			v1.x = vertex_coords.x[v1_i];
			v1.y = vertex_coords.y[v1_i];
			v2.x = vertex_coords.x[v2_i];
			v2.y = vertex_coords.y[v2_i];

			// calc centroid of edge
			//FVPoint2D<double> centroid;
			edge_centroids.x[i] = (vertex_coords.x[ edge_fst_vertex[i] ] + vertex_coords.x[ edge_snd_vertex[i] ]) * 0.5;
			edge_centroids.y[i] = (vertex_coords.y[ edge_fst_vertex[i] ] + vertex_coords.y[ edge_snd_vertex[i] ]) * 0.5;

			// calc edge length
			FVPoint2D<double> u;
			u = v1 - v2;
			edge_lengths[i] = Norm(u);
		}

		// compute centroid and perimeter for each cell
		for(unsigned int i = 0; i < num_cells; ++i) {
			cell_areas[i] = 0;
			cell_perimeters[i] = 0;
			cell_centroids.x[i] = 0;
			cell_centroids.y[i] = 0;

			for(unsigned int e = 0; e < cell_edges_count[i]; ++e) {
				unsigned int edge = cell_edges.elem(e, 0, i);

				cell_perimeters[i] += edge_lengths[edge];
				
				// add to centroid
				cell_centroids.x[i] += edge_centroids.x[edge] * edge_lengths[edge];
				cell_centroids.y[i] += edge_centroids.y[edge] * edge_lengths[edge];

				if (edge_left_cells[edge] == NO_CELL)
					edge_left_cells[edge] = i;
				else
					edge_right_cells[edge] = i;
			}
			cell_centroids.x[i] /= cell_perimeters[i];
			cell_centroids.y[i] /= cell_perimeters[i];

			// compute area of each cell
			for(unsigned int e = 0; e < cell_edges_count[i]; ++e) {
				unsigned int edge = cell_edges.elem(e, 0, i);

				// 0 is x coord
				cell_edges_normal.elem(e, 0, i) = edge_centroids.x[edge] - cell_centroids.x[i];
				// 1 is y coord
				cell_edges_normal.elem(e, 1, i) = edge_centroids.y[edge] - cell_centroids.y[i];
				//cell2edges[e].x[i] = edge_centroids.x[edge] - cell_centroids.x[i];
				//cell2edges[e].y[i] = edge_centroids.y[edge] - cell_centroids.y[i];

				FVPoint2D<double> u, v;
				u.x = vertex_coords.x[ edge_fst_vertex[edge] ] - cell_centroids.x[i];
				u.y = vertex_coords.y[ edge_fst_vertex[edge] ] - cell_centroids.y[i];

				v.x = vertex_coords.x[ edge_snd_vertex[edge] ] - cell_centroids.x[i];
				v.y = vertex_coords.y[ edge_snd_vertex[edge] ] - cell_centroids.y[i];

				cell_areas[i] += fabs(Det(u,v)) * 0.5;
			}

			// update list of vertex->cell pointer
			// TODO what is this???
		}

		for(unsigned int i = 0; i < num_edges; ++i) {
			unsigned int fst_vertex = edge_fst_vertex[i];
			unsigned int snd_vertex = edge_snd_vertex[i];

			FVPoint2D<double> u, v;
			double no;

			u.x = vertex_coords.x[fst_vertex] - vertex_coords.x[snd_vertex];
			u.y = vertex_coords.y[fst_vertex] - vertex_coords.y[snd_vertex];
			no = Norm(u);

			edge_normals.x[i] = u.y / no;
			edge_normals.y[i] = -u.x / no;
			
			v.x = edge_centroids.x[i] - cell_centroids.x[ edge_left_cells[i] ];
			v.y = edge_centroids.y[i] - cell_centroids.y[ edge_left_cells[i] ];

			double temp = edge_normals.x[i] * v.x + edge_normals.y[i] * v.y;
			if (temp < 0) {
				edge_normals.x[i] *= -1. ;
				edge_normals.y[i] *= -1. ;
			}

			if (edge_right_cells[i] == NO_CELL) {
				// TODO push boundary edges
			}
		}

	}

	/************************************************
	 * GETTERS/SETTERS
	 ***********************************************/
	CFVMesh2D_cuda* CFVMesh2D::cuda_get() {
		return cuda_mesh;
	}

	bool CFVMesh2D::cuda_is_alloc() {
		return (cuda_mesh != NULL);
	}


	/************************************************
	 * MEMORY MANAGEMENT METHODS
	 ***********************************************/

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

	void CFVMesh2D::alloc() {
		if (num_vertex <= 0 || num_edges <= 0 || num_cells <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		// alloc vertex info
		vertex_coords		= CFVPoints2D<double>(num_vertex);

		// alloc edge info
		edge_types			= CFVArray<int>(num_edges);
		edge_normals		= CFVPoints2D<double>(num_edges);
		edge_centroids		= CFVPoints2D<double>(num_edges);
		edge_lengths		= CFVArray<double>(num_edges);
		edge_fst_vertex		= CFVArray<unsigned int>(num_edges);
		edge_snd_vertex		= CFVArray<unsigned int>(num_edges);
		edge_left_cells		= CFVArray<unsigned int>(num_edges);
		edge_right_cells	= CFVArray<unsigned int>(num_edges);

		// alloc cell info
		cell_types			= CFVArray<int>(num_cells);
		cell_areas			= CFVArray<double>(num_cells);
		cell_perimeters		= CFVArray<double>(num_cells);
		cell_centroids		= CFVPoints2D<double>(num_cells);
		//cell_edges_index	= CFVArray<unsigned int>(num_cells);
		cell_edges_count	= CFVArray<unsigned int>(num_cells);
		for(unsigned int i = 0; i < MAX_EDGES_PER_CELL; ++i) {
			cell_edges = CFVMat<unsigned int>(MAX_EDGES_PER_CELL, 1, num_cells);
			cell_edges_normal = CFVMat<double>(MAX_EDGES_PER_CELL, 2, num_cells);
		}
	}
}

