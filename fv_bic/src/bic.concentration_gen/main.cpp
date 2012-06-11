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

int gen_menu(Parameters data) {
	cout << "What model do you want for " << data.initial_file << " ?" << endl << endl
			<< " 1. sinusoidal - f(x) = sin(x)" << endl
			<< " 2. half full  - f(x) = (x < 0.5) ? 1 : 0" << endl
			<< " 3. triangle   - f(x) = (x < 0.5) ? 2x : -2x + 2" << endl
			<< " 4. constant   - f(x) = 1" << endl
			<< " 5. test data  - f(x,y) = 3x + 7y - 15" << endl
			<< " 6. double sin" << endl
			<< " 7. four parts" << endl
			<< " 8. circle" << endl;

	int res;
	cin >> res;
	cout << "choose " << res << endl;
	if (res < 1 && res > 8) {
		cout << "Invalid option!" << endl;
		exit(-1);
	}

	return res;
}

#define _USE_MATH_DEFINES
#include <math.h>

void fill_polu(CFVMesh2D &mesh, CFVArray<double> &polu, int op) {
	for(unsigned int i = 0; i < mesh.num_cells; ++i) {
		double x = mesh.cell_centroids.x[i];
		double y = mesh.cell_centroids.y[i];
		switch(op) {
			case 1:
				polu[i] = sin(2 * M_PI * x);
				break;
			case 2:
				polu[i] = (x < 0.5) ? 1.0 : 0.0;
				break;
			case 3:
				polu[i] = (x < 0.5) ? 2 * x : -2 * x + 2;
				break;
			case 4:
				polu[i] = 1.0;
				break;
			case 5:
				polu[i] = 3.0 * x + 5 * y - 15;
				break;
			case 6:
				polu[i] = sin(2 * M_PI * x) * sin(2 * M_PI * y);
				break;
			case 7:
				if (x < 0.5 && y < 0.5) polu[i] = 1;
				else polu[i] = 0;
				break;
			case 8:
				if ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) <= 0.2*0.2) polu[i] = 1;
				else polu[i] = 0;
		}
	}
}

int main(int argc, char **argv) {

	// read params
	Parameters data;
	if (argc != 2) {
		cerr << "Arg warning: no xml param filename specified. Defaulting to param.xml" << endl;
		data = read_parameters("param.xml");
	} else
		data = read_parameters(argv[1]);

	double t = 0; string name("concentration");

	int op = gen_menu(data);

	CFVMesh2D mesh(data.mesh_file);
	FVXMLWriter output(data.initial_file);

	CFVArray<double> polu(mesh.num_cells);
	fill_polu(mesh, polu, op);
	output.append(polu, t, name);
	output.save();
	output.close();

	cout << "success. exiting" << endl;
}

