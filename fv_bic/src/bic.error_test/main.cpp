#include "FVL/FVLib.h"
#include "FVL/FVXMLWriter.h"
#include "FVL/FVArray.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
#include "FVL/CFVMesh2D.h"
using namespace std;
using namespace FVL;

typedef struct _parameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	double anim_time;
	int anim_jump;
	double dirichlet;
	double CFL;
} Parameters;

// TODO: interface decente para paremetros xml
Parameters read_parameters (string parameters_filename) {
	Parameters data;
	FVParameters para(parameters_filename);

	data.mesh_file		= para.getString("MeshName");
	data.velocity_file	= para.getString("VelocityFile");
	data.initial_file	= para.getString("PoluInitFile");
	data.output_file	= para.getString("OutputFile");
	data.final_time		= para.getDouble("FinalTime");
	data.anim_time		= para.getDouble("AnimTimeStep");
	data.anim_jump		= para.getInteger("NbJump");
	data.dirichlet		= para.getDouble("DirichletCondition");
	data.CFL			= para.getDouble("CFL");

	return data;
}

int main(int argc, char **argv) {

	// read params
	Parameters data;
	if (argc != 2) {
		cerr << "Arg warning: no xml param filename specified. Defaulting to param.xml" << endl;
		data = read_parameters("param.xml");
	} else
		data = read_parameters(argv[1]);


	CFVMesh2D mesh(data.mesh_file);
	double time; string name;
	FVXMLReader input(data.output_file);
	int size;
	xml_node<> *current_node = input.getRootNode()->first_node();
	FVXMLReader::str_cast<int>(size, current_node->first_attribute("size")->value());

	CFVArray<double> initial(size);
	CFVArray<double> current(size);

	// read initial value, and copy it to current
	input.getVec(initial, time, name);
	for(int i = 0; i < size; ++i)
		current[i] = initial[i];

	cout << "time\terror\terror2" << endl;

	for(; current_node != 0; current_node = current_node->next_sibling()) {
		double error = numeric_limits<double>::min();
		double error2 = numeric_limits<double>::min();

		for(int i = 0; i < size; ++i) {
			double error_i = abs(initial[i] - current[i]);
			if (error_i > error)
				error = error_i;

			error2 += error_i * mesh.cell_areas[i];
		}

		cout << time << "\t" << error << ",\t" << error2 << endl;

		input.getVec(current, time, name);
	}
}

