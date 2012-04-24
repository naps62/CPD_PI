#include "partitioner.h"

#include <set>
#include <map>

void distribute_cells(FVMesh2D_SOA &mesh, vector<PartitionData> &partitions) {
	/* for each cell index, it's x coord */
	map<double, set<unsigned int> > ordered_cells;
	map<double, set<unsigned int> >::iterator map_it;
	set<unsigned int>::iterator set_it;

	// create ordered set of all cells
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		ordered_cells[ mesh.cell_centroids.x[cell] ].insert(cell);
	}

	map_it = ordered_cells.begin();
	set_it = map_it->second.begin();

	// calc num_cells for each partition
	unsigned int cells_per_part	= mesh.num_cells / partitions.size();
	unsigned int rest_cells		= mesh.num_cells % partitions.size();		
	for(unsigned int i = 0; i < partitions.size(); ++i) {
		unsigned int num_cells = cells_per_part;
		if (rest_cells > 0) {
			num_cells++;
			rest_cells--;
		}

		// create vector of cell indexes for this partition
		for(unsigned int current_count = 0; current_count < num_cells; ++current_count) {
			partitions[i].cells.insert(*set_it);
			set_it++;
			if (set_it == map_it->second.end()) {
				map_it++;
				set_it = map_it->second.begin();
			}
		}
	}
}

void distribute_edges(FVMesh2D_SOA &mesh, vector<PartitionData> &partitions) {

	// for each partition, create list of edges required
	for(vector<PartitionData>::iterator part_it = partitions.begin(); part_it != partitions.end(); ++part_it) {
		
		// for each cell in the partition, add the required edges to the set
		for(set<unsigned int>::iterator cell_it = part_it->cells.begin(); cell_it != part_it->cells.end(); ++cell_it) {

			// iterate edges of a cell, and add them to the set
			for(unsigned int i = 0; i < MAX_EDGES_PER_CELL; ++i)
				part_it->edges.insert( mesh.cell_edges.elem(i, 0, *cell_it) );
		}
	}
}

void alloc_partitions(FVMesh2D_SOA &mesh, FVArray<double> &v, vector<PartitionData> &partitions, vector<FVMesh2D_SOA_Lite *> &result) {

	// alloc data and initialize cells_left and edges_left counters
	for(unsigned int part_i = 0; part_i < partitions.size(); ++part_i) {
		partitions[part_i].edges_current = 0;
		partitions[part_i].cells_current = 0;
		result.push_back(new FVMesh2D_SOA_Lite(partitions[part_i].edges.size(), partitions[part_i].cells.size()));
	}

	// assign each edge to the right partition. an edge can belong to more than one partition
	for(unsigned int edge_i = 0; edge_i < mesh.num_edges; ++edge_i) {

		// for each partition, check it this edge belongs to it
		for(unsigned int part_data_i = 0; part_data_i < partitions.size(); ++part_data_i) {
			PartitionData & part_data	= partitions[part_data_i];
			FVMesh2D_SOA_Lite * part 	= result[part_data_i];

			// if edge belongs to this partition
			set<unsigned int>::iterator edge_it = part_data.edges.find(edge_i);
			if (edge_it != part_data.edges.end()) {
				part->edge_index		[part_data.edges_current] = *edge_it;
				part->edge_lengths		[part_data.edges_current] = mesh.edge_lengths[*edge_it];
				part->edge_velocity		[part_data.edges_current] = v[*edge_it];
				part->edge_left_cells	[part_data.edges_current] = mesh.edge_left_cells[*edge_it];
				part->edge_right_cells	[part_data.edges_current] = mesh.edge_right_cells[*edge_it];

				// fix left-rigth cells, if necessary
				// that is, if right cell exists and left cell is the one on another partition
				// then swap left and right cells
				if (part->edge_right_cells[part_data.edges_current] != NO_RIGHT_CELL && part_data.cells.find( part->edge_left_cells[part_data.edges_current] ) == part_data.cells.end()) {
					unsigned int tmp = part->edge_left_cells[part_data.edges_current];
					part->edge_left_cells [part_data.edges_current] = part->edge_right_cells[part_data.edges_current];
					part->edge_right_cells[part_data.edges_current] = tmp;
				}
				part_data.edges_current++;
			}
		}

	}

	// assign each cell to the right partition. each cell belongs only to a single partition
	for(unsigned int cell_i = 0; cell_i < mesh.num_cells; ++cell_i) {

		// for each partition, check if this edge belongs to it
		for(unsigned int part_data_i = 0; part_data_i < partitions.size(); ++part_data_i) {
			PartitionData & part_data	= partitions[part_data_i];
			FVMesh2D_SOA_Lite * part	= result[part_data_i];

			// if cell belongs to this partition
			set<unsigned int>::iterator cell_it = part_data.cells.find(cell_i);
			if (cell_it != part_data.cells.end()) {
				part->cell_index		[part_data.cells_current] = *cell_it;
				part->cell_areas		[part_data.cells_current] = mesh.cell_areas[*cell_it];
				part->cell_edges_count	[part_data.cells_current] = mesh.cell_edges_count[*cell_it];

				// copy list of edges
				for(unsigned int e = 0; e < part->cell_edges_count[part_data.cells_current]; ++e) {
					part->cell_edges.elem(e, 0, part_data.cells_current) = mesh.cell_edges.elem(e, 0, *cell_it);
				}
				part_data.cells_current++;
			}
		}
	}

	// fill partition neighbors data
	for(unsigned int part_i = 0; part_i < result.size(); ++part_i) {
		PartitionData & part_data = partitions[part_i];
		FVMesh2D_SOA_Lite * part  = result[part_i];

		for(unsigned int e = 0; e < part->num_edges; ++e) {
			unsigned int cell = part->edge_right_cells[e];

			// if cell exists
			if (cell != NO_RIGHT_CELL) {
				// if cell exists in current partition, nothing to do here
				if (part_data.cells.find(cell) != part_data.cells.end()) {
					part->edge_part[e] = 0;
				}

				// if cell exists in left partition
				else if (part_i > 0 && partitions[part_i - 1].cells.find(cell) != partitions[part_i - 1].cells.end()) {
					part->edge_part[e] = -1;
					part->edge_part_index[e] = part->left_cells++;
				}

				// by exclusion, it can only exist in the right partition
				else {
					part->edge_part[e] = 1;
					part->edge_part_index[e] = part->right_cells++;
				}
			}
			else {
				part->edge_part[e] = 0;
			}
		}
	}

	// fix edge and cell indexing, to be relative to the partition and not the global mesh
	for(unsigned int part_i = 0; part_i < result.size(); ++part_i) {
		FVMesh2D_SOA_Lite * part = result[part_i];

		// fix edges
		for(unsigned int e = 0; e < part->num_edges; ++e) {
			unsigned int edge_val = part->edge_index[e];

			// fix cell_edges
			for(unsigned int c = 0; c < part->num_cells; ++c) {
				for(unsigned int e2 = 0; e2 < part->cell_edges_count[c]; ++e2) {
					if (part->cell_edges.elem(e2, 0, c) == edge_val)
						part->cell_edges.elem(e2, 0, c) = e;
				}
			}
		}

		// fix cells
		for(unsigned int c = 0; c < part->num_cells; ++c) {
			unsigned int cell_val = part->cell_index[c];

			// fix edge_left_cell, and some right cells
			for(unsigned int e = 0; e < part->num_edges; ++e) {
				if (part->edge_left_cells[e] == cell_val)
					part->edge_left_cells[e] = c;
				else if (part->edge_right_cells[e] == cell_val)
					part->edge_right_cells[e] = c;
			}
		}

		// fill index_to_edge arrays
		part->left_index_to_edge  = new CFVArray<unsigned int>(part->left_cells);
		part->right_index_to_edge = new CFVArray<unsigned int>(part->right_cells);
		for(unsigned int e = 0; e < part->num_edges; ++e) {
			// if this edge is linked to another partition
			switch(part->edge_part[e]) {
				case -1:
					part->left_index_to_edge[0][ part->edge_part_index[e] ] = e;
					break;
				case 1:
					part->right_index_to_edge[0][ part->edge_part_index[e] ] = e;
					break;
				default: // nothing to do here
					break;
			}
		}
	}
}

void generate_partitions(FVMesh2D_SOA &mesh, FVArray<double> &velocity, int num_partitions, vector<FVMesh2D_SOA_Lite *> &result) {

	/* this struct will hold paramteres of each partition while generating them */
	vector<PartitionData> partitions(num_partitions);

	distribute_cells(mesh, partitions);
	distribute_edges(mesh, partitions);
	alloc_partitions(mesh, velocity, partitions, result);
}
