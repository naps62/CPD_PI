#include <fstream>
#include <string>

#include "FVL/CUDA/CFVMesh2D.h"
#include "FVLib_config.h"
#include "FVPoint2D.h"

#include "rapidxml/rapidxml.hpp"
#include "FVL/FVio.h"
#include "FVL/FVXMLReader.h"
#include "FVL/FVErr.h"

using namespace rapidxml;

namespace FVL {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	CFVMesh2D::CFVMesh2D() {
	}

	CFVMesh2D::CFVMesh2D(FVMesh2D &msh) {
		import_FVMesh2D(msh);
	}

	CFVMesh2D::CFVMesh2D(const string &filename) {
		read_mesh_file(filename);
	}

	CFVMesh2D::~CFVMesh2D() {
	}

	/************************************************
	 * IMPORT/EXPORT METHODS
	 ***********************************************/

	void CFVMesh2D::import_FVMesh2D(FVMesh2D &msh) {
		num_edges = msh.getNbEdge();
		num_cells = msh.getNbCell();

		//cout << "Importing mesh: " << num_cells << " cells, " << num_edges << " edges" << endl;
		
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
			edge_right_cells[i]	= (edge->rightCell != NULL) ? (edge->rightCell->label - 1) : NO_RIGHT_EDGE; 
		}

		// copy cell data
		// caso nao haja disponibilidade da sua partei = 0;
		FVCell2D *cell;
		i = 0;
		//num_total_edges = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ++i) {
			// cell area
			cell_areas[i]	= cell->area;

			// index at which edges for this cell start
			//cell_edges_index[i] = num_total_edges;
			// count of edges for this cell
			cell_edges_count[i] = cell->nb_edge;

			// total count of edges for cell_edges array
			//num_total_edges += cell->nb_edge;
		}

		// finally create data for cell_edges array
		// this is not in alloc() func since it depends on values calculated on previous loop

		//cell_edges = CFVVect<unsigned int>(num_total_edges);
		i = 0;
		for(msh.beginCell(); (cell = msh.nextCell()); ) {
			for(cell->beginEdge(); (edge = cell->nextEdge()); ++i) {
				cell_edges[i] = edge->label - 1;
			}
		}

		/*cout << "num_total_edges = "<< num_total_edges << endl;
		int j = 0;
		for(i = 0; i < num_cells; ++i) {
			cout << "cell " << i << " at " << j << " with " << cell_edges_count[i] << "edges:\t";
			for(j = 0; j < cell_edges_count[i]; ++j) {
				cout << cell_edges[ cell_edges_index[i] + j ] << "     ";
			}
			cout << endl;
		}
		exit(0);*/
	}

	void CFVMesh2D::read_mesh_file(const string &filename) {
		//xml_document<> mesh;
		//vector<char> xml;
		//string file(filename);
		//FVio::parse_xml(mesh, xml, file);
		FVXMLReader mesh(filename);
		// TODO testar esta alteração (acima)

		// get reference for each element list
		cout << mesh.first_node()->name() << endl;
		xml_node<> *vertex	= mesh.first_node()->first_node()->first_node();
		xml_node<> *edge	= vertex->next_sibling();
		xml_node<> *cell	= edge->next_sibling();
		cout << vertex->first_attribute()->name() << endl;

		// get count of each element
		FVio::str_cast<unsigned int>(num_vertex, vertex->first_attribute("nbvertex")->value());
		FVio::str_cast<unsigned int>(num_edges, edge->first_attribute("nbedge")->value());
		FVio::str_cast<unsigned int>(num_cells, cell->first_attribute("nbcell")->value());

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
			edge_ss >> type;		// dummy, unused
			edge_ss >> cell_count;	// dummy, unused
			edge_ss >> vertex1;		// x coord
			edge_ss >> vertex2;		// y coord

			edge_fst_vertex[i] = vertex1 - 1;
			edge_snd_vertex[i] = vertex2 - 1;
		}

		// read cell data
		stringstream cell_ss(cell->value());
		for(unsigned int i = 0; i < num_cells; ++i) {
			int id, type;

			cell_ss >> id;					// read id
			id--;							// change id to 0-based index
			cell_ss >> type;				// dummy, unused
			cell_ss >> cell_edges_count[i];	// number of edges on this cell

			if (cell_edges_count[i] > MAX_EDGES_PER_CELL) {
				string msg("edges per cell exceed MAX_EDGES_PER_CELL. please update flag and recompile");
				FVErr::error(msg, 1);
			}

			for(unsigned int e = 0; e < cell_edges_count[i]; ++e) {
				cell_ss >> cell_edges[e][i];	// reads e'th edge of cell i
				cell_edges[e][i]--;				// change to 0-based index
			}
		}

		this->compute_final_data();
	}

	void CFVMesh2D::compute_final_data() {
		// initialize vertex
		// compute centroid and length
		for(unsigned int i = 0; i < num_edges; ++i) {
			edge_left_cells[i] = NO_EDGE;
			edge_right_cells[i] = NO_EDGE;

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
				unsigned int edge = cell_edges[e][i];

				cell_perimeters[i] += edge_lengths[edge];
				
				// add to centroid
				cell_centroids.x[i] += edge_centroids.x[edge] * edge_lengths[i];
				cell_centroids.y[i] += edge_centroids.y[edge] * edge_lengths[i];

				if (edge_left_cells[edge] == NO_EDGE)
					edge_left_cells[edge] = i;
				else
					edge_right_cells[edge] = i;
			}
			cell_centroids.x[i] /= cell_perimeters[i];
			cell_centroids.y[i] /= cell_perimeters[i];

			// compute area of each cell
			for(unsigned int e = 0; e < cell_edges_count[i]; ++e) {
				unsigned int edge = cell_edges[e][i];

				cell2edges[e].x[i] = edge_centroids.x[edge] - cell_centroids.x[i];
				cell2edges[e].y[i] = edge_centroids.y[edge] - cell_centroids.y[i];

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

			if (edge_right_cells[i] == NO_EDGE) {
				// TODO push boundary edges
			}
		}

	}


	/************************************************
	 * MEMORY MANAGEMENT METHODS
	 ***********************************************/
	void CFVMesh2D::alloc() {
		if (num_vertex <= 0 || num_edges <= 0 || num_cells <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		// alloc vertex info
		vertex_coords		= CFVPoints2D(num_vertex);

		// alloc edge info
		edge_normals		= CFVPoints2D(num_edges);
		edge_centroids		= CFVPoints2D(num_edges);
		edge_lengths		= CFVVect<double>(num_edges);
		edge_fst_vertex		= CFVVect<unsigned int>(num_edges);
		edge_snd_vertex		= CFVVect<unsigned int>(num_edges);
		edge_left_cells		= CFVVect<unsigned int>(num_edges);
		edge_right_cells	= CFVVect<unsigned int>(num_edges);

		// alloc cell info
		cell_areas			= CFVVect<double>(num_cells);
		cell_perimeters		= CFVVect<double>(num_cells);
		cell_centroids		= CFVPoints2D(num_cells);
		//cell_edges_index	= CFVVect<unsigned int>(num_cells);
		cell_edges_count	= CFVVect<unsigned int>(num_cells);
		for(unsigned int i = 0; i < MAX_EDGES_PER_CELL; ++i) {
			cell_edges.push_back(CFVVect<unsigned int>(num_cells));
			cell2edges.push_back(CFVPoints2D(num_cells));
		}
	}
}

