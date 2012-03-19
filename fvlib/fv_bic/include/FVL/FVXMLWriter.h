/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLWriter.h
** XML Writer class based on RapidXML
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#ifndef _H_FVXMLWRITER
#define _H_FVXMLWRITER

#include <sstream>
#include <iomanip>
using namespace std;

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "FVL/CFVVect.h"
#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"
using namespace rapidxml;

namespace FVL {
	class FVXMLWriter : public xml_document<> {

		private:
			string filename;
			ofstream out_stream;

			xml_node<> *root;

		public:
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			FVXMLWriter();
			FVXMLWriter(string filename);

			void init();

			void save();
			void save(string filename);
			void close();


			/************************************************
			 * OUTPUT (methods will be added as needed)
			 ***********************************************/

			// intended for output frame of an animation
			template<class T> void append(CFVVect<T> &vec, double time=0.0, string name="noname");

		private:
			// aux functions to allocate and append attributes
			template<class T> void add_attribute(xml_node<> *node, string name, T value);
	};

}

#include "FVL/templates/FVXMLWriter.hpp"

#endif // _H_FVXMLWRITER
