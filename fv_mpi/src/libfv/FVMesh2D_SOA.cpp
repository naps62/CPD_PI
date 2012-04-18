#include <fstream>
#include <string>
#include <set>
#include <map>

#include "FVL/FVMesh2D_SOA.h"
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

	FVMesh2D_SOA::FVMesh2D_SOA(unsigned int num_vertex, unsigned int num_edges, unsigned int num_cells) {
		this->num_vertex = num_vertex;
		this->num_edges	 = num_edges;
		this->num_cells  = num_cells;

		vertex_cells = CFVArray<double>(0);

		alloc();
	}

	FVMesh2D_SOA::FVMesh2D_SOA(FVMesh2D &msh) {
		import_FVMesh2D(msh);
	}

	FVMesh2D_SOA::FVMesh2D_SOA(const string &filename) {
		read_mesh_file(filename);
	}

	/************************************************
	 * IMPORT/EXPORT METHODS
	 ***********************************************/

	void FVMesh2D_SOA::import_FVMesh2D(FVMesh2D &msh) {
		num_vertex	= msh.getNbVertex();
		num_edges	= msh.getNbEdge();
		num_cells	= msh.getNbCell();

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

	void FVMesh2D_SOA::read_mesh_file(const string &filename) {
		FVXMLReader mesh(filename);

		// get reference for each element list
		xml_node<> *vertex	= mesh.first_node()->first_node()->first_node();
		xml_node<> *edge	= vertex->next_sibling();
		xml_node<> *cell	= edge->next_sibling();

		// get count of each element
		FVXMLReader::str_cast<unsigned int>(num_vertex, vertex->first_attribute("nbvertex", 0, false)->value());
		FVXMLReader::str_cast<unsigned int>(num_edges, edge->first_attribute("nbedge", 0, false)->value());
		FVXMLReader::str_cast<unsigned int>(num_cells, cell->first_attribute("nbcell", 0, false)->value());

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

			set<unsigned int> tmp_cell_vertexes;

			for(unsigned int e = 0; e < cell_edges_count[i]; ++e) {
				unsigned int edge;
				cell_ss >> edge;	// reads e'th edge of cell i
				edge--;				// change to 0-based index
				cell_edges.elem(e, 0, i) = edge;	

				tmp_cell_vertexes.insert(edge_fst_vertex[edge]);
				tmp_cell_vertexes.insert(edge_snd_vertex[edge]);
			}

			int v = 0;
			set<unsigned int>::iterator it;
			for(it = tmp_cell_vertexes.begin(); it != tmp_cell_vertexes.end(); ++it, ++v) {
				cell_vertexes.elem(v, 0, i) = *it;
			}
			for(unsigned int v = 0; v < cell_edges_count[i]; ++v) {
			}
		}

		this->compute_final_data();
	}

	void FVMesh2D_SOA::compute_final_data() {
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
		}

		// update list of vertex->cell pointer
		map<unsigned int, set<unsigned int> > tmp_vertex_cells;

		// create a temporary map of (vertex, set(cells)) that map))
		for(unsigned int cell = 0; cell < num_cells; ++cell) {
			for(unsigned int edge_i = 0; edge_i < cell_edges_count[cell]; ++edge_i) {
				unsigned int edge = cell_edges.elem(edge_i, 0, cell);
				tmp_vertex_cells[edge_fst_vertex[edge]].insert(cell);
				tmp_vertex_cells[edge_snd_vertex[edge]].insert(cell);
			}
		}

		// computes total size of the created map
		unsigned int total_count = 0;
		for(unsigned int vertex = 0; vertex < num_vertex; ++vertex) {
			vertex_cells_index[vertex] = total_count;
			vertex_cells_count[vertex] = tmp_vertex_cells[vertex].size();
			total_count += vertex_cells_count[vertex];
		}

		// converts the map to a sequential array
		vertex_cells = CFVArray<double>(total_count);
		unsigned int counter = 0;
		for(unsigned int vertex = 0; vertex < num_vertex; ++vertex) {
			set<unsigned int>::iterator cells_it;
			for(cells_it = tmp_vertex_cells[vertex].begin(); cells_it != tmp_vertex_cells[vertex].end(); ++cells_it) {
				vertex_cells[counter++] = *cells_it;
			}
		}
	}

	/************************************************
	 * MEMORY MANAGEMENT METHODS
	 ***********************************************/
	void FVMesh2D_SOA::alloc() {
		if (num_vertex <= 0 || num_edges <= 0 || num_cells <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		// alloc vertex info
		vertex_coords		= CFVPoints2D<double>(num_vertex);
		vertex_cells_count	= CFVArray<double>(num_vertex);
		vertex_cells_index	= CFVArray<double>(num_vertex);

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
			cell_vertexes = CFVMat<unsigned int>(MAX_VERTEX_PER_CELL, 1, num_cells);
		}
	}
}

