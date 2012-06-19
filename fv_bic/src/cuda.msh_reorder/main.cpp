#include "FVL/FVMesh2D_SOA.h"
#include "FVL/FVParameters.h"
using namespace std;
using namespace FVL;

/**
 * Parameters struct passed via xml file
 */
struct Parameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	double anim_time;
	int anim_jump;
	double dirichlet;
	double CFL;

	public:
	// Constructor receives parameter file
	Parameters(string parameters_filename) {
		FVL::FVParameters para(parameters_filename);

		this->mesh_file		= para.getString("MeshName");
		this->velocity_file	= para.getString("VelocityFile");
		this->initial_file	= para.getString("PoluInitFile");
		this->output_file	= para.getString("OutputFile");
		this->final_time	= para.getDouble("FinalTime");
		this->anim_time		= para.getDouble("AnimTimeStep");
		this->anim_jump		= para.getInteger("NbJump");
		this->dirichlet		= para.getDouble("DirichletCondition");
		this->CFL			= para.getDouble("CFL");
	}
};

void swap_cell_edge_reference(FVMesh2D_SOA &mesh, int cell, int l, int r) {
	if (cell == NO_RIGHT_CELL)
		return;

	for(int edge_i = mesh.cell_edges_count[cell] - 1; edge_i >= 0; --edge_i) {
		if (mesh.cell_edges.elem(edge_i, 0, cell) == l) {
			mesh.cell_edges.elem(edge_i, 0, cell) = r;
			return;
		}
	}
}

template<class T> void _swap(T &x, T &y) {
	T tmp = x;
	x = y;
	y = tmp;
}

// swaps edge l with edge r, updating indexes on all cells that reference it
void swap_edges(FVMesh2D_SOA &mesh, int l, int r) {
	swap_cell_edge_reference(mesh, mesh.edge_left_cells[l],  l, r);
	swap_cell_edge_reference(mesh, mesh.edge_right_cells[l], l, r);
	swap_cell_edge_reference(mesh, mesh.edge_left_cells[r],  r, l);
	swap_cell_edge_reference(mesh, mesh.edge_right_cells[r], r, l);

	_swap<unsigned int>(mesh.edge_fst_vertex[l],  mesh.edge_fst_vertex[r]);
	_swap<unsigned int>(mesh.edge_snd_vertex[l],  mesh.edge_snd_vertex[r]);
	_swap<unsigned int>(mesh.edge_left_cells[l],  mesh.edge_left_cells[r]);
	_swap<unsigned int>(mesh.edge_right_cells[l], mesh.edge_right_cells[r]);
}

int main(int argc, char **argv) {
	string param_filename;
	if (argc != 2) {
		param_filename = "param.xml";
	} else
		param_filename = argv[1];

	Parameters data(param_filename);
	
	// read mesh
	FVL::FVMesh2D_SOA mesh(data.mesh_file);
	
	unsigned int l = 0;					 // iterates from the left, finding edges that HAVE a right cell
	unsigned int r = mesh.num_edges - 1; // iterates from the rigth, finding edges that DONT HAVE a right cell

	// until the two iterators meet
	while(l < r) {

		// find next left edge WITH a right cell
		while(mesh.edge_right_cells[l] == NO_RIGHT_CELL) ++l;
		while(mesh.edge_right_cells[r] != NO_RIGHT_CELL) --r;

		swap_edges(mesh, l, r);
	}

	mesh.save("mesh.ordered.xml");
}
