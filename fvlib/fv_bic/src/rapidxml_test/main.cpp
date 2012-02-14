// Test of rapidxml library

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "FVL/FVio.h"
#include "FVL/CFVVect.h"
#include "FVL/FVXMLWriter.h"
#include "FVL/FVXMLReader.h"
#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"
using namespace std;
using namespace rapidxml;


int main() {

	FVL::FVXMLWriter writer;
	FVL::CFVVect<double> vec(10);
	for(int i = 0; i < 10; i++)
		vec[i] = i*2;
	writer.append(vec);
	cout << writer << endl;
/*	FVL::FVXMLReader reader("xml_samples/foz.xml");

	xml_node<>* vertex = reader.first_node()->first_node()->first_node();
	cout << "vertex node: " << vertex->name() << endl;
	exit(0);*/

	/*xml_document<> doc;
	vector<char> vec;
	string filename("xml_samples/foz.xml");
	FVL::FVio::parse_xml(doc, vec, filename);
	xml_node<> *vertex = doc.first_node();
	cout << "vertex node: " << vertex->name() << endl;*/
	

	/*ifstream mesh_file("xml_samples/foz.xml");
	string mesh_xml;
	string line;

	while(getline(mesh_file, line))
		mesh_xml += line;

	// make a safe-to-modify copy
	vector<char> xml_copy(mesh_xml.begin(), mesh_xml.end());
	xml_copy.push_back('\0');

	*/
/*	xml_document<> mesh;
	vector<char> xml;
	//mesh.parse<0>(&xml_copy[0]);
	string file("xml_samples/foz.xml");
	MFVio::parse_xml(mesh, xml, file); 
	
	xml_node<>* vertex	= mesh.first_node()->first_node()->first_node();
	xml_node<>* edge	= vertex->next_sibling();
	xml_node<>*	cell	= edge->next_sibling();

	cout << "vertex node: " << vertex->name() << endl;
	cout << "edge node: " << edge->name() << endl;
	cout << "cell node: " << cell->name() << endl;

	int vertexC;
	MFVio::str_cast<int>(vertexC, vertex->first_attribute("nbvertex")->value());

	cout << "num vertex: " << vertexC + 2 << endl;
	cout << "num edge: " << edge->first_attribute("nbedge")->value() << endl;
	cout << "num cell: " << cell->first_attribute("nbcell")->value() << endl;

	//stringstream ss;w 
	//print(cout, *vertex, 0);
	
	//cout << "vertex data: " << vertex->value() << endl;
	
	int max_edges;
	MFVio::str_cast<int>(max_edges, edge->first_attribute("nbedge")->value());
	stringstream ss(edge->value());
	for(unsigned int i = 0; i < max_edges; ++i) {
		int id;
		int type;
		int cell_count;
		int left_cell;
		int right_cell;

		ss >> id;
		ss >> type;
		ss >> cell_count;
		ss >> left_cell;
		ss >> right_cell;

		//cout << id << " " << type << " " << cell_count << " " << left_cell << " " << right_cell << endl;
	}*/
}
