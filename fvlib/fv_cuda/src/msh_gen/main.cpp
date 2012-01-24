/**
 * Author: Miguel Palhas
 * mpalhas@gmail.com
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace std;

// TODO for now only SQUARE IS IMPLEMENTED
enum MESH_TYPE {
	SQUARE,
	TRIANGLE,
	REVERSE_TRIANGLE,
	CROSS,
};

class Vertex {
	unsigned int id;
	unsigned int type;
	unsigned int x;
	unsigned int y;

	public:
	Vertex(unsigned int newid, int newtype, int newx, int newy)
		{  id=newid; type=newtype; x=newx; y=newy; }

	friend ostream& operator << (ostream &stream, const Vertex &v);
};

ostream& operator << (ostream &stream, const Vertex &v) {
	stream
		<< "\t\t\t" << v.id		// vertex label
		<< "\t"		<< v.type	// vertex type
		<< "\t"		<< v.x		// x coord	
		<< "\t"		<< v.y	 	// y coord
		<< endl;
	return stream;
}

class Edge {
	unsigned int id;
	unsigned int type;
	vector<unsigned int> lower_dim_list;

	public:
	Edge(unsigned int newid, unsigned int newtype, vector<unsigned int> new_lower_dim_list)
		{ id=newid; type=newtype; lower_dim_list=new_lower_dim_list; }

	friend ostream& operator << (ostream &stream, const Edge &e);
};

ostream& operator << (ostream &stream, const Edge &e) {
	stream
		<< "\t\t\t" << e.id							// edge/cell label
		<< "\t"		<< e.type						// edge/cell type
		<< "\t"		<< e.lower_dim_list.size(); 	// vertex/edge count
		// print vertex/edge list
	for(unsigned int i = 0; i < e.lower_dim_list.size(); ++i)
		stream << "\t" << e.lower_dim_list[i];
	stream << endl;
	return stream;
}

typedef Edge Cell;


int main(int argc, char **argv) {
	if (argc != 6) {
		cout
			<< "Invalid usage. Correct parameters: " << endl
			<< "msg_gen file_name Lx Ly wX wY" << endl;
		exit(-1);
	}

	// argument read
	string name 	= string(argv[1]);
	unsigned int Lx = atoi(argv[2]);
	unsigned int Ly = atoi(argv[3]);
	unsigned int Wx = atoi(argv[4]);
	unsigned int Wy = atoi(argv[5]);

	unsigned int num_vertex	= (Lx +1) * (Ly + 1);
	unsigned int num_edges	= (Lx * Ly * 2) + Lx + Ly;
	unsigned int num_cells	= Lx * Ly;
	
	ofstream		output(name.c_str());
	stringstream 	vertex_stream;
	stringstream	edge_stream;
	stringstream	cell_stream;

	unsigned int i_vertex	= 1;
	unsigned int i_edge		= 1;
	unsigned int i_cell		= 1;

	unsigned int x, y;
	// create first line of vertexes
	for(x = 0; x <= Lx; ++x) {
		vertex_stream << Vertex(i_vertex++, 0, x * Wx, 0);
	}

	// create first line of edges
	for(x = 1; x <= Lx; ++x) {
		vector<unsigned int> vertex_list;
		vertex_list.push_back(x);
		vertex_list.push_back(x+1);
		edge_stream << Edge(i_edge++, 0, vertex_list);
	}

	for(y = 0; y < Ly; ++y) {
		// Next line of vertex
		for(x = 0; x <= Lx; ++x) {
			vertex_stream << Vertex(i_vertex++, 0, x * Wx, (y+1) * Wy);
		}

		// Vertical edges
		for(x = 0; x <= Lx; ++x) {
			unsigned int v1 = (Lx + 1) * y + x + 1;
			unsigned int v2 = v1 + Lx + 1;
			vector<unsigned int> vertex_list;
			vertex_list.push_back(v1);
			vertex_list.push_back(v2);
			edge_stream << Edge(i_edge++, 0, vertex_list);
		}

		// y'th line of horizontal edges (not counting top one) and cells
		for(x = 1; x <= Lx; ++x) {
			// edge
			unsigned int v1 = (Lx + 1) * (y + 1) + x;
			unsigned int v2 = v1 + 1;
			vector<unsigned int> vertex_list;
			vertex_list.push_back(v1);
			vertex_list.push_back(v2);
			edge_stream << Edge(i_edge++, 0, vertex_list);

			// cell
			unsigned int e1 = y * Lx + y * (Lx + 1) + x;
			unsigned int e2 = e1 + Lx;
			unsigned int e3 = e2 + 1;
			unsigned int e4 = e3 + Lx;
			vector<unsigned int> edge_list;
			edge_list.push_back(e1);
			edge_list.push_back(e2);
			edge_list.push_back(e3);
			edge_list.push_back(e4);
			cell_stream << Cell(i_cell++, 0, edge_list);
		}
	}

	output
		<< "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" 	<< endl
		<< "<FVLIB>" 											<< endl
		<< "\t<MESH dim=\"2\"	name=\"fvcm convertor\">" 		<< endl
		<< "\t\t<VERTEX nbvertex=\""	<< num_vertex << "\">" 	<< endl
		<< vertex_stream.str()									
		<< "\t\t</VERTEX>" 										<< endl
		<< "\t\t<EDGE nbedge=\"" 		<< num_edges << "\">"	<< endl
		<< edge_stream.str()									
		<< "\t\t</EDGE>"										<< endl
		<< "\t\t<CELL nbcell=\""		<< num_cells << "\">"	<< endl
		<< cell_stream.str()									
		<< "\t\t</CELL>"										<< endl
		<< "\t</MESH>"											<< endl
		<< "</FVLIB>"											<< endl;

	output.close();
}
