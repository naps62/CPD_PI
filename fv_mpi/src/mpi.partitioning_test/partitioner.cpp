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

#define foreach(C, x, y) for(C::iterator x = y.begin(); x != y.end(); ++y)

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

void alloc_partitions(FVMesh2D_SOA &result, vector<PartitionData> &partitions, vector<FVMesh2D_SOA_Lite> &result) {
	/* allocate each partition */
	unsigned int part = 0
	for(vector<PartitionData>::iterator it = partitions.begin(); it != partitions.end(); ++it) {
		result.push_back(CFVMesh2D_SOA_Lite(0, it->num_edges, it->num_cells));

		/* save cell data for this part */
		for(unsigned int cell = 0; cell < it->num_cells; ++cell) {
			result.back().cell_centroids.x[cell] = mesh.cell_centroids.x[ it->cells[cell] ];
			result.back().cell_centroids.y[cell] = mesh.cell_centroids.y[ it->cells[cell] ];
		}

		for(unsigned int edge = 0; edge < it->num_edges; ++edge) {

		}
	}


}

void generate_partitions(FVMesh2D_SOA &mesh, int num_partitions, vector<FVMesh2D_SOA_Lite> &result) {

	/* this struct will hold paramteres of each partition while generating them */
	vector<PartitionData> partitions(num_partitions);

	distribute_cells(mesh, partitions);
	distribute_edges(mesh, partitions);

	alloc_partitions(partitions, result);

	/*for(int i = 0; i < partitions.size(); ++i)
		partitions[i].dump();

	return vector<FVMesh2D_SOA>();*/
}
