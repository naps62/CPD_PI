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

void alloc_partitions(FVMesh2D_SOA &mesh, FVArray<double> &v, vector<PartitionData> &partitions, FVMesh2D_SOA_Lite * &result, int id) {

	// alloc data and initialize cells_left and edges_left counters
	unsigned int edges_current = 0;
	unsigned int cells_current = 0;
	PartitionData &part_data = partitions[id];
	result = new FVMesh2D_SOA_Lite(part_data.edges.size(), part_data.cells.size());

	for(set<unsigned int>::iterator edge_it = part_data.edges.begin(); edge_it != part_data.edges.end(); ++edge_it) {
		unsigned int edge = *edge_it;

		result->edge_index		[edges_current] = edge;
		result->edge_lengths	[edges_current] = mesh.edge_lengths[edge];
		result->edge_velocity	[edges_current] = v[edge];
		result->edge_left_cells	[edges_current] = mesh.edge_left_cells[edge];
		result->edge_right_cells[edges_current] = mesh.edge_right_cells[edge];

		// fix left-rigth cells, if necessary
		// that is, if right cell exists and left cell is the one on another partition
		// then swap left and right cells
		if (result->edge_right_cells[edges_current] != NO_RIGHT_CELL && part_data.cells.find( result->edge_left_cells[edges_current] ) == part_data.cells.end()) {
			unsigned int tmp = result->edge_left_cells[edges_current];
			result->edge_left_cells [edges_current] = result->edge_right_cells[edges_current];
			result->edge_right_cells[edges_current] = tmp;
		}
		part_data.edges_current++;
	}

	for(set<unsigned int>::iterator cell_it = part_data.cells.begin(); cell_it != part_data.cells.end(); ++cell_it) {
		unsigned int cell = *cell_it;

		result->cell_index		[cells_current] = cell;
		result->cell_areas		[cells_current] = mesh.cell_areas[cell];
		result->cell_edges_count[cells_current] = mesh.cell_edges_count[cell];

		// copy list of edges
		for(unsigned int e = 0; e < result->cell_edges_count[cells_current]; ++e) {
			result->cell_edges.elem(e, 0, cells_current) = mesh.cell_edges.elem(e, 0, cell);
		}
		cells_current++;
	}

	// fill partition neighbors data
	for(unsigned int edge = 0; edge < result->num_edges; ++edge) {
		unsigned int cell = result->edge_right_cells[edge];

		// if cell doesnt exist or if it exists in current partition, nothing to do here
		if (cell == NO_RIGHT_CELL || part_data.cells.find(cell) != part_data.cells.end()) {
			result->edge_part[edge] = 0;
		}

		// if cell exists in left partition
		else if (id > 0 && partitions[id - 1].cells.find(cell) != partitions[id - 1].cells.end()) {
			result->edge_part[edge]		= -1;
			result->edge_part_index[edge] = result->left_cells++;
		}

		// by exclusion, it can only exist in the right partition
		else {
			result->edge_part[edge] = 1;
			result->edge_part_index[edge] = result->right_cells++;
		}
	}

	// fix edge and cell indexing, to be relative to the partition and not the global mesh

   // fix edges
	for(unsigned int e = 0; e < result->num_edges; ++e) {
		unsigned int edge_val = result->edge_index[e];

		// fix cell_edges
		for(unsigned int c = 0; c < result->num_cells; ++c) {
			for(unsigned int e2 = 0; e2 < result->cell_edges_count[c]; ++e2) {
				if (result->cell_edges.elem(e2, 0, c) == edge_val)
					result->cell_edges.elem(e2, 0, c) = e;
			}
		}
	}

	// fix cells
	for(unsigned int c = 0; c < result->num_cells; ++c) {
		unsigned int cell_val = result->cell_index[c];
	
		// fix edge_left_cell, and some right cells
		for(unsigned int e = 0; e < result->num_edges; ++e) {
			if (result->edge_left_cells[e] == cell_val)
				result->edge_left_cells[e] = c;
			else if (result->edge_right_cells[e] == cell_val)
				result->edge_right_cells[e] = c;
		}
	}

	// fill index_to_edge arrays
	result->left_index_to_edge  = new CFVArray<unsigned int>(result->left_cells);
	result->right_index_to_edge = new CFVArray<unsigned int>(result->right_cells);
	for(unsigned int e = 0; e < result->num_edges; ++e) {
		// if this edge is linked to another partition
		switch(result->edge_part[e]) {
			case -1:
				result->left_index_to_edge[0][ result->edge_part_index[e] ] = e;
				break;
			case 1:
				result->right_index_to_edge[0][ result->edge_part_index[e] ] = e;
				break;
			default: // nothing to do here
				break;
		}
	}
}

void generate_partitions(FVMesh2D_SOA &mesh, FVArray<double> &velocity, int id, int size, FVMesh2D_SOA_Lite* &result) {

	/* this struct will hold paramteres of each partition while generating them */
	vector<PartitionData> partitions(size);

	distribute_cells(mesh, partitions);
	distribute_edges(mesh, partitions);
	alloc_partitions(mesh, velocity, partitions, result, id);
}
