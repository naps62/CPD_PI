#include "FVL/FVLib.h"
#include "FVL/FVXMLWriter.h"
#include "FVL/FVArray.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
#include "FVL/FVMesh2D_SOA.h"
#include "kernels.h"

using namespace std;
using namespace FVL;

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

double compute_mesh_parameter(FVL::FVMesh2D_SOA mesh) {
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
		unsigned int left	= mesh.edge_left_cells[i];
		unsigned int right	= mesh.edge_right_cells[i];

		if (right == NO_RIGHT_CELL)
			right = left;

		double v	= ((velocities.x[left] + velocities.x[right]) * 0.5 * mesh.edge_normals.x[i])
			+ ((velocities.y[left] + velocities.y[right]) * 0.5 * mesh.edge_normals.y[i]);

		vs[i] = v;

		if (abs(v) > v_max || i == 0) {
			v_max = abs(v);
		}
	}
}

int main (int argc, char **argv) {

	// var declaration
	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	Parameters data;
	if (argc != 2) {
		cerr << "Arg warning: no xml param filename specified. Defaulting to param.xml" << endl;
		data = read_parameters("param.xml");
	} else
		data = read_parameters(argv[1]);

#ifdef PROFILE_LIMITED
	if (!id)
		PROFILE_INIT();
#endif

	// read mesh
	FVL::FVMesh2D_SOA mesh(data.mesh_file);

	FVL::CFVPoints2D<double> velocities(mesh.num_cells);
	FVL::FVArray<double> polution(mesh.num_cells);
	FVL::FVArray<double> flux(mesh.num_edges);
	FVL::FVArray<double> vs(mesh.num_edges);

	// read other input files
	FVL::FVXMLReader velocity_reader(data.velocity_file);
	FVL::FVXMLReader polu_ini_reader(data.initial_file);
	polu_ini_reader.getVec(polution, t, name);
	velocity_reader.getPoints2D(velocities, t, name);
	polu_ini_reader.close();
	velocity_reader.close();

	FVL::FVXMLWriter polution_writer(data.output_file);
	polution_writer.append(polution, t, "polution");
	//polution.dump();
	//cout << "end polution" << endl;

	// compute velocity vector
	compute_edge_velocities(mesh, velocities, vs, v_max);
	h = compute_mesh_parameter(mesh);
	dt	= 1.0 / v_max * h;

	//for(unsigned int edge = 0; edge < mesh.num_edges; ++edge)
		//cout << " edge " << edge << " v " << vs[edge] << " left " << setw(3) << mesh.edge_left_cells[edge] << " right " << setw(3) << mesh.edge_right_cells[edge] <<  endl;

#ifdef PROFILE1
	if (!id)
		PROFILE_START();
#endif
#ifdef PROFILE_LIMITED
	for (mliters = 0; mliters < PROFILE_LIMITED; ++mliters)
#else
	while(t < data.final_time)
#endif
	{
		cout << "iteration " << i << '\r';

		compute_flux(mesh, vs, polution, flux, data.dirichlet);
		update(mesh, polution, flux, dt);

		t += dt;
		if (i % data.anim_jump == 0)
			polution_writer.append(polution, t, "polution");
		++i;
	}
#ifdef PROFILE1
	if (!id)
		PROFILE_STOP();
#endif

	//for(unsigned int j = 0; j < mesh.num_cells; ++j)
	//		cout << "polution[" << j << "] = " << polution[j] << endl;
	
	cout << endl << "finished" << endl;

	polution_writer.append(polution, t, "polution");
	polution_writer.save();
	polution_writer.close();

#ifdef PROFILE1
	if (!id) {
		PROFILE_OUTPUT();
		PROFILE_CLEANUP();
	}
#endif
}

