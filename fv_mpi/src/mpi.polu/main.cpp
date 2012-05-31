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

int id, size;

#ifdef PROFILE_LIMITED
	long unsigned mliters;
#endif

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

double compute_mesh_parameter(FVMesh2D_SOA &mesh) {
	double h;
	double S;

	h = 1.e20;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		S = mesh.cell_areas[cell];
		for(unsigned int edge = 0; edge < mesh.cell_edges_count[cell]; ++edge) {
			double length = mesh.edge_lengths[edge];
			if (h * length > S)
				h = S / length;
		}
	}

	return h;
}

void compute_edge_velocities(FVMesh2D_SOA &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max) {
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		unsigned int left = mesh.edge_left_cells[i];
		unsigned int right = mesh.edge_right_cells[i];

		if (right == NO_RIGHT_CELL)
			right = left;

		double v = ((velocities.x[left] + velocities.x[right]) * 0.5 * mesh.edge_normals.x[i])
				 + ((velocities.y[left] + velocities.y[right]) * 0.5 * mesh.edge_normals.y[i]);

		vs[i] = v;
		if (abs(v) > v_max || i == 0)
			v_max = abs(v);
	}
}


void dump_partition(FVL::FVMesh2D_SOA_Lite &part) {
	for(int i = 0; i < size; ++i) {
		if (id == i)
			part.left_index_to_edge->dump();
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void append_anim(FVL::FVXMLWriter &writer, string name, double t, FVL::FVMesh2D_SOA_Lite &partition, FVL::CFVArray<double> &global_polu, int num_procs) {
	// slaves send their polution values to master
	if (id != 0) {
		MPI_Send(&partition.num_cells,     1,                   MPI_INT,    0, TAG_WRITER_SIZE,  MPI_COMM_WORLD);// first send cell count
		MPI_Send(&partition.cell_index[0], partition.num_cells, MPI_INT,    0, TAG_WRITER_INDEX, MPI_COMM_WORLD);// then send cell_index array
		MPI_Send(&partition.polution[0],   partition.num_cells, MPI_DOUBLE, 0, TAG_WRITER_POLU,  MPI_COMM_WORLD);// finally send polution values
	} else {
		// master receives all and outputs to writer

		// copies it's own polution to global array
		for(unsigned int x = 0; x < partition.num_cells; ++x)
			global_polu[ partition.cell_index[x] ] = partition.polution[x];

		unsigned int count;
		MPI_Status status;
		FVL::CFVArray<unsigned int>* tmp_cell_index;
		FVL::CFVArray<double>* tmp_polu;
		for(int i = 1; i < num_procs; ++i) {
			MPI_Recv(&count, 1, MPI_INT, i, TAG_WRITER_SIZE, MPI_COMM_WORLD, &status);					// recv array size
			//cout << count << endl;
			tmp_cell_index = new FVL::CFVArray<unsigned int>(count);
			tmp_polu       = new FVL::CFVArray<double>(count);

			MPI_Recv(&tmp_cell_index[0][0], count, MPI_INT,    i, TAG_WRITER_INDEX, MPI_COMM_WORLD, &status);	// recv cell_index
			MPI_Recv(&tmp_polu[0][0],       count, MPI_DOUBLE, i, TAG_WRITER_POLU,  MPI_COMM_WORLD, &status);		// recv polu

			// save polu to global array
			for(unsigned int x = 0; x < count; ++x)
				global_polu[ tmp_cell_index[0][x] ] = tmp_polu[0][x];

			delete tmp_cell_index;
			delete tmp_polu;
		}

		writer.append(global_polu, t, name);
	}
}

int main(int argc, char **argv) {

	int i = 1;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	Parameters data;
	if (argc != 2) {
		cerr << "Arg warning: no xml param filename specified. Defaulting to param.xml and 2 nodes" << endl;
		data = read_parameters("param.xml");
	} else {
		data = read_parameters(argv[1]);
	}

	//MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef PROFILE
		PROFILE_INIT();
#endif

	// read mesh
	FVL::FVMesh2D_SOA mesh(data.mesh_file);
	FVL::CFVPoints2D<double> velocities(mesh.num_cells);
	FVL::CFVArray<double> polution(mesh.num_cells);
	FVL::CFVArray<double> vs(mesh.num_edges);

	FVL::FVXMLReader velocity_reader(data.velocity_file);
	FVL::FVXMLReader polu_ini_reader(data.initial_file);
	polu_ini_reader.getVec(polution, t, name);
	velocity_reader.getPoints2D(velocities, t, name);
	polu_ini_reader.close();
	velocity_reader.close();


	FVL::FVXMLWriter polution_writer;
	FVL::FVArray<double> global_polu;
	if (id == 0) {
		polution_writer.open(data.output_file);
		global_polu = FVL::FVArray<double>(mesh.num_cells);
	}

	compute_edge_velocities(mesh, velocities, vs, v_max);
	h = compute_mesh_parameter(mesh);
	dt = 1.0 / v_max * h;

	// PARTITION THE MESH
	vector<PartitionData> part_data(size);
	generate_partitions(mesh, id, size, part_data);
	FVL::FVMesh2D_SOA_Lite partition(part_data[id].edges.size(), part_data[id].cells.size());
	alloc_partitions(mesh, vs, polution, part_data, partition, id);

	FVL::CFVArray<double> flux(partition.num_edges);

	//	main loop
	//append_anim(polution_writer, "polution", t, partition, global_polu, size);
#ifdef PROFILE_LIMITED
	for (mliters = 0; mliters < PROFILE_LIMITED; ++mliters)
#else
	while (t < data.final_time)
#endif
	{
		communication(id, size, partition, polution);
		compute_flux(partition, flux, data.dirichlet, id);
		update(partition, flux, dt);

		t += dt;
		if (i % data.anim_jump == 0) {
			if (!id)
				cerr << "Animation frame " << i / data.anim_jump << endl;
			append_anim(polution_writer, "polution", t, partition, global_polu, size);
		}
		++i;
	}

	append_anim(polution_writer, "polution", t, partition, global_polu, size);

	if (id == 0) {
		polution_writer.save();
		polution_writer.close();
		cout << endl << "finished" << endl;
	}

#ifdef PROFILE
		PROFILE_OUTPUT();
		PROFILE_CLEANUP();
#endif

	MPI_Finalize();
}

