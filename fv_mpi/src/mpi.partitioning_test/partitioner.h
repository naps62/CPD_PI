#ifndef _H_PARTITIONER
#define _H_PARTITIONER

#include <vector>
#include <set>
#include "FVL/FVMesh2D_SOA.h"
using namespace FVL;
using namespace std;

struct PartitionData {
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

vector<FVMesh2D_SOA> generate_partitions(FVMesh2D_SOA &mesh, int num_partitions);

#endif // _H_PARTITIONER
