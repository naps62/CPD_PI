/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** Filename: FVXMLReader.hpp
 ** XML Reader class based on RapidXML
 **
 ** Author:		Miguel Palhas, mpalhas@gmail.com
 ** Created:	13-02-2012
 ** Last Test:	---
 ** -------------------------------------------------------------------------*/

#ifdef _H_FVXMLREADER

#ifndef _HPP_FVXMLREADER
#define _HPP_FVXMLREADER

#include <sstream>
#include <iomanip>
#include <string>
using namespace std;

#include "FVL/CFVArray.h"
#include "FVL/CFVPoints2D.h"
#include "rapidxml/rapidxml.hpp"
using namespace rapidxml;

namespace FVL {

	template<class T>
		bool FVXMLReader::str_cast(T &t, const string &s) {
			istringstream iss(s);
			return !(iss >> t).fail();
		}

	// Reads next child node to a vector from the XML file
	template<class T>
		void FVXMLReader::getVec(CFVArray<T> &vec, double &time, string &name) {
			xml_attribute<> *time_attr = current->first_attribute("time", 0, false);
			xml_attribute<> *name_attr = current->first_attribute("name", 0, false);

			FVXMLReader::str_cast<double>(time, time_attr->value());
			FVXMLReader::str_cast<string>(name, name_attr->value());

			// read vector
			stringstream ss(current->value());
			for(unsigned int i = 0; i < vec.size(); ++i) {
				ss >> vec[i];
			}

			// current now points to next sibling of xml file
			current = current->next_sibling();
		}

	// Reads next child node to a vector of 2D points from the XML file
	template<class T>
		void FVXMLReader::getPoints2D(CFVPoints2D<T> &vec, double &time, string &name) {
			xml_attribute<> *time_attr = current->first_attribute("time", 0, false);
			xml_attribute<> *name_attr = current->first_attribute("name", 0, false);

			FVXMLReader::str_cast<double>(time, time_attr->value());
			FVXMLReader::str_cast<string>(name, name_attr->value());

			// read vector
			stringstream ss(current->value());
			for(unsigned int i = 0; i < vec.size(); ++i) {
				ss >> vec.x[i];
				ss >> vec.y[i];
			}

			// current now points to next sibling of xml file
			current = current->next_sibling();
		}
}

#endif // _HPP_FVXMLREADER
#endif // _H_FVXMLREADER
