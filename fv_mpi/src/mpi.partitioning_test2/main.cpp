#include "FVL/FVLib.h"
#include "FVL/FVXMLWriter.h"
#include "FVL/FVArray.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
#include "FVL/FVMesh2D_SOA.h"
#include "FVL/FVMesh2D_SOA_Lite.h"
#include "kernels.h"

#include "partitioner.h"

#include <mpi.h>

using namespace std;
using namespace FVL;

struct Parameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	int anim_jump;
	double dirichlet;
	double CFL;
};

Parameters read_parameters (string parameters_filename) {
	Parameters data;
	FVL::FVParameters para(parameters_filename);

	data.mesh_file		= para.getString("MeshName");
	data.velocity_file	= para.getString("VelocityFile");
	data.initial_file	= para.getString("PoluInitFile");
	data.output_file	= para.getString("OutputFile");
	data.final_time		= para.getDouble("FinalTime");
	data.anim_jump		= para.getInteger("NbJump");
	data.dirichlet		= para.getDouble("DirichletCondition");
	data.CFL			= para.getDouble("CFL");

	return data;
}

int id, size;

void dump_partition(FVL::FVMesh2D_SOA_Lite * &part) {
	for(int i = 0; i < size; ++i) {
		if (id == i)
			part->cell_index.dump();
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

//void mpi_polu_simulation(FVMesh2D_SOA_Lite * &partitions, Parameters &data, FVXMLWriter &polu_writer) {
//}

int main(int argc, char **argv) {

	// read params
	Parameters data;
	if (argc != 3) {
		cerr << "Arg warning: no xml param filename specified. Defaulting to param.xml and 2 nodes" << endl;
		data = read_parameters("param.xml");
	} else {
		data = read_parameters(argv[1]);
	}

	//MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// read mesh
	FVL::FVMesh2D_SOA mesh(data.mesh_file);
	FVL::FVArray<double> velocities(mesh.num_edges);

	FVL::FVMesh2D_SOA_Lite* partition;
	generate_partitions(mesh, velocities, id, size, partition);

	dump_partition(partition);

	//mpi_polu_simulation(partition);
}

