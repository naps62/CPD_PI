#ifndef _H_PARTITIONER
#define _H_PARTITIONER

#include <vector>
#include <set>
#include "FVL/FVMesh2D_SOA.h"
#include "FVL/FVMesh2D_SOA_Lite.h"
using namespace FVL;
using namespace std;

struct PartitionData {
	unsigned int cells_current;
	unsigned int edges_current;
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
void alloc_partitions(FVMesh2D_SOA &mesh, FVArray<double> &v, vector<PartitionData> &partitions, vector<FVMesh2D_SOA_Lite *> &result);

void generate_partitions(FVMesh2D_SOA &mesh, FVArray<double> &velocity, int id, int size, FVMesh2D_SOA_Lite * &result);

#endif // _H_PARTITIONER
