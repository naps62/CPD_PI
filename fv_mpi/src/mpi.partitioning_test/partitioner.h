#ifndef _H_PARTITIONER
#define _H_PARTITIONER

#include <vector>
#include <set>
#include "FVL/FVMesh2D_SOA.h"
using namespace FVL;
using namespace std;

struct PartitionData {
	unsigned int cells_left;
	unsigned int edges_left;
	set<unsigned int> cells;
	set<unsigned int> edges;

	void dump() {
		cout << "partition: " << endl;
		dump_array(cells);
		dump_array(edges);
	}

	void dump_array(set<unsigned int> &arr) {
		cout << "[";
		for(set<unsigned int>::iterator it = arr.begin(); it != arr.end(); ++it)
			cout << *it << ", ";
		cout << "]" << endl;
	}
};

void distribute_cells(FVMesh2D_SOA &mesh, vector<PartitionData> &partitions);
void distribute_edges(FVMesh2D_SOA &mesh, vector<PartitionData> &partitions);
vector<FVMesh2D_SOA> alloc_partitions(vector<PartitionData> &partitions, vector<FVMesh2D_SOA_Lite> &result);

vector<FVMesh2D_SOA> generate_partitions(FVMesh2D_SOA &mesh, int num_partitions, vector<FVMesh2D_SOA_Lite> &result);

#endif // _H_PARTITIONER
