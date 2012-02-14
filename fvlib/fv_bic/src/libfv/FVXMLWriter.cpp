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
#include <sstream>
using std::stringstream;

#include "FVL/FVXMLWriter.h"

#include "rapidxml/rapidxml_print.hpp"

namespace FVL {

	FVXMLWriter::FVXMLWriter() {
		init();
	}

	void FVXMLWriter::init() {
		// append xml declaration
		xml_node<> *decl = this->allocate_node(node_declaration);
		decl->append_attribute(this->allocate_attribute("version", "1.0"));
		decl->append_attribute(this->allocate_attribute("encoding", "utf-8"));
		this->append_node(decl);

		// initial FVLIB node
		xml_node<> *root = this->allocate_node(node_element, "FVLIB");
		this->append_node(root);
	}

	
}
