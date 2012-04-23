#include "partitioner.h"

#include <set>
#include <map>

void distribute_cells(FVMesh2D_SOA &mesh, vector<PartitionData> &partitions) {
	/* for each cell index, it's x coord */
	map<double, set<unsigned int> > ordered_cells;
	map<double, set<unsigned int> >::iterator map_it;
	set<unsigned int>::iterator set_it;

	/* create ordered set of all cells */
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		ordered_cells[ mesh.cell_centroids.x[cell] ].insert(cell);
	}

	map_it = ordered_cells.begin();
	set_it = map_it->second.begin();

	/* calc num_cells for each partition */
	unsigned int cells_per_part	= mesh.num_cells / partitions.size();
	unsigned int rest_cells		= mesh.num_cells % partitions.size();		
	for(unsigned int i = 0; i < partitions.size(); ++i) {
		unsigned int num_cells = cells_per_part;
		if (rest_cells > 0) {
			num_cells++;
			rest_cells--;
		}

		/* create vector of cell indexes for this partition */
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

	/* for each partition, create list of edges required */
	for(vector<PartitionData>::iterator part_it = partitions.begin(); part_it != partitions.end(); ++part_it) {
		
		/* for each cell in the partition, add the required edges to the set */
		for(set<unsigned int>::iterator cell_it = part_it->cells.begin(); cell_it != part_it->cells.end(); ++cell_it) {

			/* iterate edges of a cell, and add them to the set */
			for(unsigned int i = 0; i < MAX_EDGES_PER_CELL; ++i)
				part_it->edges.insert( mesh.cell_edges.elem(i, 0, *cell_it) );
		}
	}
}

void alloc_partitions(FVMesh2D_SOA &mesh, FVArray<double> &v, vector<PartitionData> &partitions, vector<FVMesh2D_SOA_Lite *> &result) {

	/* allocate each partition */
	unsigned int current_part = 0;
	for(unsigned current_part = 0; current_part < partitions.size(); ++current_part) {
		PartitionData part_data = partitions[current_part];
	//for(vector<PartitionData>::iterator it = partitions.begin(); it != partitions.end(); ++it, ++current_part) {
		result.push_back(new FVMesh2D_SOA_Lite(part_data.edges.size(), part_data.cells.size()));

		FVMesh2D_SOA_Lite part = *(result.back());

		/* save cell data for this part */
		unsigned int cell_i = 0;
		for(set<unsigned int>::iterator cell_it = part_data.cells.begin(); cell_it != part_data.cells.end(); ++cell_it, ++cell_i) {
			unsigned int cell = *cell_it;

			part.cell_index[cell_i] 		= cell;
			part.cell_areas[cell_i] 		= mesh.cell_areas[cell];
			part.cell_edges_count[cell_i]	= mesh.cell_edges_count[cell];
			
			/* copy edge list for each cell */
			for(unsigned int edge_i = 0; edge_i < mesh.cell_edges_count[cell]; ++edge_i) {
				part.cell_edges.elem(edge_i, 0, cell_i) = mesh.cell_edges.elem(edge_i, 0, cell);
			}
		}

		/* save edge data */
		unsigned int edge_i = 0;
		for(set<unsigned int>::iterator edge_it = part_data.edges.begin(); edge_it != part_data.edges.end(); ++edge_it, ++edge_i) {
			unsigned int edge = *edge_it;

			part.edge_index[edge_i]			= edge;
			part.edge_lengths[edge_i]		= mesh.edge_lengths[edge];
			part.edge_velocity[edge_i]		= v[edge];
			part.edge_left_cells[edge_i]	= mesh.edge_left_cells[edge_i];
			part.edge_right_cells[edge_i]	= mesh.edge_right_cells[edge_i];
		}

		/* save edge partition reference */
		for(unsigned int edge = 0; edge < part->num_edges; 
	}

	/* save edge partition references */
	unsigned int current_part = 0;
	unsigned int left_index = 0, right_index = 0;
	for(vector<FVMesh2D_SOA_Lite *>::iterator it = result.begin(); it != result.end(); ++result, ++current_part) {
		FVMesh2D_SOA_Lite part = *it;
		for(unsigned int edge = 0; edge < part.num_edges; ++edge) {
			unsigned int cell;

			/* if left cell is not in the current mesh, swap left and right cells, to mantain mesh rules (left cell must always exist */
			cell = mesh.edge_left_cells[edge];
			if (part_data.cells.find(l_cell) == part_data.cells.end()) {
				mesh.edge_left_cells[edge]	= mesh.edge_right_cells[edge];
				mesh.edge_right_cells[edge] = cell;
				cell = mesh.edge_left_cell[edge];
			}
	
			/* if right edge exists, checks its partition */
			if (part.edge_types[edge] != FV_EDGE_DIRICHLET) {
				/*  cell exists in current partition */
				if (part_data.cells.find(cell) != part_data.cells.end())
					part.edge_left_part = 0;
				/* else if it exists in left partition */
				else if (current_part > 0 && 				 partitions[current_part - 1].cells.find(cell) != part_data.cells.end()) {
					part.edge_left_part = -1;
					part.edge_left_part_index = left_index++;
				}
				else if (current_part < partitions.size() && partitions[current_part + 1].cells.find(cell) != part_data.cells.end()) {
					part.edge_left_part =  1;
					part.edge_right_part_index = right_index++;
				}
				else
					cout << "error finding right cell " << cell << " in partition " << current_part << endl;
			}
		}
	}
}

void generate_partitions(FVMesh2D_SOA &mesh, int num_partitions, vector<FVMesh2D_SOA_Lite *> &result) {

	/* this struct will hold paramteres of each partition while generating them */
	vector<PartitionData> partitions(num_partitions);

	distribute_cells(mesh, partitions);
	distribute_edges(mesh, partitions);
	alloc_partitions(mesh, partitions, result);
}
