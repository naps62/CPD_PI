/**
 * Author: Miguel Palhas
 * mpalhas@gmail.com
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "FVL/FVGlobal.h"
using namespace std;
using namespace FVL;

// TODO for now only SQUARE IS IMPLEMENTED
enum MESH_TYPE {
	SQUARE,
	TRIANGLE,
	REVERSE_TRIANGLE,
	CROSS,
};

class Vertex {
	public:
	unsigned int id;
	unsigned int type;
	double x;
	double y;

	Vertex(unsigned int newid, int newtype, double newx, double newy)
		{  id=newid; type=newtype; x=newx; y=newy; }

	friend ostream& operator << (ostream &stream, const Vertex &v);
};

ostream& operator << (ostream &stream, const Vertex &v) {
	stream
		<< "\t\t\t" << v.id		// vertex label
		<< "\t"		<< v.type	// vertex type
		<< "\t"		<< scientific << setprecision(FV_PRECISION) << setw(FV_CHAMP) << v.x		// x coord	
		<< "\t"		<< scientific << setprecision(FV_PRECISION) << setw(FV_CHAMP) << v.y	 	// y coord
		<< endl;
	return stream;
}

class Edge {
	public:
	unsigned int id;
	unsigned int type;
	vector<unsigned int> lower_dim_list;

	Edge(unsigned int newid, unsigned int newtype, vector<unsigned int> new_lower_dim_list)
		{ id=newid; type=newtype; lower_dim_list=new_lower_dim_list; }

	Edge(unsigned int newid, unsigned int newtype, unsigned int list1, unsigned int list2)
		{ id=newid; type=newtype; lower_dim_list.push_back(list1); lower_dim_list.push_back(list2); }

	Edge(unsigned int newid, unsigned int newtype, unsigned int list1, unsigned int list2, unsigned int list3, unsigned int list4)
		{ id=newid; type=newtype; lower_dim_list.push_back(list1); lower_dim_list.push_back(list2); lower_dim_list.push_back(list3); lower_dim_list.push_back(list4); }

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
	if (argc < 4) {
		cout
			<< "Invalid usage. Correct parameters: " << endl
			<< "msg_gen file_name X Y" << endl;
		exit(-1);
	}

	// argument read
	string name 	= string(argv[1]);
	double cX = strtod(argv[2], NULL);	
	double cY = strtod(argv[3], NULL);
	double vX = cX + 1;
	//double vY = cY + 1;

	double cW = 1 / cX;

	if (argc == 5)
		cW = atoi(argv[4]) / cX;

	//unsigned int num_cells	= cX * cY;
	//unsigned int num_vertex	= vX * vY;
	//unsigned int num_edges  = num_vertex + num_cells;
	
	ofstream		output(name.c_str());
	stringstream 	vertex_stream;
	stringstream	edge_stream;
	stringstream	cell_stream;


#define VERTEX(x,y) (vertex_stream		 << Vertex(++i_vertex, 0,   (x), (y)))
#define EDGE(t,x,y) (edge_stream 		 <<   Edge(++i_edge,   (t), (x), (y)))
#define CELL(t,x1,x2,x3,x4) (cell_stream <<   Cell(++i_cell,   (t), (x1), (x2), (x3), (x4)))

	unsigned int i_vertex	= 0;
	unsigned int i_edge		= 0;
	unsigned int i_cell		= 0;

	unsigned int x, y;

	// create first line of vertexes (edge vertex)
	for(x = 0; x < vX; ++x) VERTEX(x * cW, 0.0);

	// create first line of edges
	for(x = 0; x < cX; ++x)	EDGE(2, x+1, x+2);

	for(y = 0; y < cY; ++y) {
		// Next line of vertex (first and last are tagged as edge vertexes
		//int edge_type = (y == cY - 1) ? 2 : 0;
		for(x = 0; x < vX; ++x) {
			VERTEX(x * cW, (y+1) * cW);
		}

		// Vertical edges
		for(x = 0; x <= cX; ++x) {
			int edge_type = (x == 0 || x == cX) ? 1 : 0;
			unsigned int v1 = vX * y + x + 1;
			unsigned int v2 = v1 + vX;
			EDGE(edge_type, v1, v2);
		}

		// y'th line of horizontal edges and cells
		// (last line of edge is edge-type)
		int edge_type = (y == cY - 1) ? 2 : 0;
		for(x = 0; x < cX; ++x) {

			// edge
			unsigned int v1 = vX * (y+1) + x + 1;
			unsigned int v2 = v1 + 1;
			EDGE(edge_type, v1, v2);

			// cell
			unsigned int e1 = y * cX + y * vX + x + 1;
			unsigned int e2 = e1 + cX;
			unsigned int e3 = e2 + 1;
			unsigned int e4 = e3 + cX;
			CELL(0, e1, e2, e3, e4);
		}
	}

	output
		<< "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" 	<< endl
		<< "<FVLIB>" 											<< endl
		<< "\t<MESH dim=\"2\"	name=\"fvcm convertor\">" 		<< endl
		<< "\t\t<VERTEX nbvertex=\""	<< i_vertex << "\">" 	<< endl
		<< vertex_stream.str()									
		<< "\t\t</VERTEX>" 										<< endl
		<< "\t\t<EDGE nbedge=\"" 		<< i_edge << "\">"		<< endl
		<< edge_stream.str()									
		<< "\t\t</EDGE>"										<< endl
		<< "\t\t<CELL nbcell=\""		<< i_cell << "\">"		<< endl
		<< cell_stream.str()									
		<< "\t\t</CELL>"										<< endl
		<< "\t</MESH>"											<< endl
		<< "</FVLIB>"											<< endl;

	output.close();
}
