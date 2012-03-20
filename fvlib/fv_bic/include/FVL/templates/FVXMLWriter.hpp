/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLWriter.hpp
** XML Writer class based on RapidXML
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#ifdef _H_FVXMLWRITER

#ifndef _HPP_FVXMLWRITER
#define _HPP_FVXMLWRITER

#include <sstream>
#include <iomanip>
using namespace std;

#include "FVL/CFVArray.h"
#include "rapidxml/rapidxml.hpp"
using namespace rapidxml;

namespace FVL {

	template<class T>
	void FVXMLWriter::append(CFVArray<T> &vec, double time, string name) {
		stringstream ss;
		ss << endl;
		ss << scientific << setprecision(FV_PRECISION) << setw(FV_CHAMP);
		for(unsigned int i = 0; i < vec.size(); ++i) {
			ss << vec[i] << " ";
		}
		ss << endl;
		char* value_str = this->allocate_string(ss.str().c_str());
		xml_node<> *node = this->allocate_node(
							node_element,
							this->allocate_string("FIELD"));
		
		// append attributes
		unsigned int nbvec = 1;
		this->add_attribute<unsigned int>	(node, string("size"), vec.size());
		this->add_attribute<unsigned int>	(node, string("nbvec"), nbvec);
		this->add_attribute<double>			(node, string("time"), time);
		this->add_attribute<string>			(node, string("name"), name);

		node->append_node(this->allocate_node(node_data, NULL, value_str));

		root->append_node(node);
	}

	template<class T>
	void FVXMLWriter::add_attribute(xml_node<> *node, string name, T value) {
		stringstream ss;
		ss << scientific << setprecision(FV_PRECISION) << setw(FV_CHAMP) << value;
		string str_value(ss.str());
		node->append_attribute(
				this->allocate_attribute(
					this->allocate_string(name.c_str()),
					this->allocate_string(str_value.c_str())));
	}
}

#endif // _HPP_FVXMLWRITER
#endif // _H_FVXMLWRITER
