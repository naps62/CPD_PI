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

//void mpi_polu_simulation(FVMesh2D_SOA_Lite * &partitions, Parameters &data, FVXMLWriter &polu_writer) {
//}

void append_anim(FVL::FVXMLWriter &writer, string name, double t, FVL::FVMesh2D_SOA_Lite &partition, FVL::CFVArray<double> &global_polu, int num_procs) {
	// slaves send their polution values to master
	if (id != 0) {
		MPI_Send(&partition.num_cells, 1, MPI_INT, 0, TAG_WRITER_SIZE, MPI_COMM_WORLD);							// first send cell count
		MPI_Send(&partition.cell_index[0], partition.num_cells, MPI_INT, 0, TAG_WRITER_INDEX, MPI_COMM_WORLD);// then send cell_index array
		MPI_Send(&partition.polution[0], partition.num_cells, MPI_DOUBLE, 0, TAG_WRITER_POLU, MPI_COMM_WORLD);// finally send polution values
	} else {
		// copies it's own polution to global array
		for(unsigned int x = 0; x < partition.num_cells; ++x)
			global_polu[ partition.cell_index[x] ] = partition.polution[x];


		// master receives all and outputs to writer
		unsigned int count;

		MPI_Status status;
		for(int i = 1; i < num_procs; ++i) {
			MPI_Recv(&count, 1, MPI_INT, i, TAG_WRITER_SIZE, MPI_COMM_WORLD, &status);					// recv array size
			FVL::CFVArray<unsigned int> tmp_cell_index(count), tmp_polu(count);

			MPI_Recv(&tmp_cell_index[0], count, MPI_INT, i, TAG_WRITER_INDEX, MPI_COMM_WORLD, &status);	// recv cell_index
			MPI_Recv(&tmp_polu[0], count, MPI_DOUBLE, i, TAG_WRITER_POLU, MPI_COMM_WORLD, &status);		// recv polu

			// save polu to global array
			for(unsigned int x = 0; x < count; ++x)
				global_polu[ tmp_cell_index[x] ] = tmp_polu[x];
		}

		//writer.append(global_polu, t, name);
	}
}

int main(int argc, char **argv) {

	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

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
	vector<double> global_polu;
	if (id == 0) {
		//polution_writer = FVL::FVXMLWriter(data.output_file);
		global_polu = FVL::FVArray<double>(mesh.num_cells);
	}

	polution.dump();
	cout << "end polution" << endl;

	compute_edge_velocities(mesh, velocities, vs, v_max);
	h = compute_mesh_parameter(mesh);
	dt = 1.0 / v_max * h;

	// watch out for velocities and polution read, need to revert velocity when rigth cell is swaped TODO

	// PARTITION THE MESH
	vector<PartitionData> part_data(size);
	generate_partitions(mesh, id, size, part_data);
	FVL::FVMesh2D_SOA_Lite partition(part_data[id].edges.size(), part_data[id].cells.size());
	alloc_partitions(mesh, vs, polution, part_data, partition, id);

	FVL::CFVArray<double> flux(partition.num_edges);

	//for(unsigned int edge = 0; edge < partition.num_edges; ++edge)
	//	cout << id << " edge " << setw(3) << edge << " global " << setw(3) << partition.edge_index[edge] << setw(2) << " part " << partition.edge_part[edge] <<  " v " << partition.edge_velocity[edge] << " left " << partition.cell_index[partition.edge_left_cells[edge]] << " right " << setw(3) << (partition.edge_right_cells[edge] == NO_RIGHT_CELL ? NO_RIGHT_CELL : partition.cell_index[partition.edge_right_cells[edge]]) << " other part " << endl;

	while (t < data.final_time) {
		if (id == 0) cout << endl << "iteration " << i << endl;
		communication(id, size, partition, polution);

		compute_flux(partition, flux, data.dirichlet);

		update(partition, flux, dt);

		cout.flush();
		MPI_Barrier(MPI_COMM_WORLD);

		t += dt;
		if (i % data.anim_jump == 0)
			append_anim(polution_writer, "polution", t, partition, global_polu, size);
			//polution_writer.append(polution, t, "polution");
		++i;
	}
		//for(unsigned int j = 0; j < partition.num_cells; ++j)
		//	cout << i << " " << id  << " polution[" << partition.cell_index[j] << "] = " << partition.polution[j] << endl;

	append_anim(polution_writer, "polution", t, partition, global_polu, size);
	if (id == 0)
		polution_writer.close();

	//mpi_polu_simulation(partition);
}

