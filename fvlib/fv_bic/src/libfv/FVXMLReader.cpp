/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLReader.cpp
** XML Reader class based on RapidXML
**
** Author: Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#include "FVL/FVXMLReader.h"

#include <fstream>
using std::ifstream;

namespace FVL {

	FVXMLReader::FVXMLReader(string filename) {
		string file_content = read_file(filename);
		xml_data = vector<char>(file_content.begin(), file_content.end());
		xml_data.push_back('\0');
		this->parse<0>(&xml_data[0]);

		// get FVLIB node
		root = this->first_node("fvlib", 0, false);
		current = root->first_node();
	}

	xml_node<>* FVXMLReader::getRootNode() {
		return root;
	}

	string FVXMLReader::read_file(string filename) {
		ifstream mesh_file(filename.c_str());
		string content;
		string line;

		while(getline(mesh_file, line))
			content += line;

		return content;
	}


	void FVXMLReader::close() {
		this->clear();
	}
}
