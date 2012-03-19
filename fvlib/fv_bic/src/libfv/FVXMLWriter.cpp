/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLWriter.cpp
** XML Writer class based on RapidXML
**
** Author: Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <sstream>
using std::stringstream;
using std::ofstream;

#include "FVL/FVXMLWriter.h"

namespace FVL {

	FVXMLWriter::FVXMLWriter() {
		this->filename = "";
		init();
	}

	FVXMLWriter::FVXMLWriter(string filename) {
		this->filename = filename;
		init();
	}

	void FVXMLWriter::init() {
		// append xml declaration
		xml_node<> *decl = this->allocate_node(node_declaration);
		decl->append_attribute(this->allocate_attribute("version", "1.0"));
		decl->append_attribute(this->allocate_attribute("encoding", "utf-8"));
		this->append_node(decl);

		// initial FVLIB node
		root = this->allocate_node(node_element, "FVLIB");
		this->append_node(root);
	}

	void FVXMLWriter::save() {
		this->save(this->filename);
	}

	void FVXMLWriter::save(string filename) {
		ofstream out(filename.c_str());
		out << *this;
		cout << *this;
	}

	void FVXMLWriter::close() {
		this->clear();
	}
	
}
