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
#include "FVL/CFVArray.h"
#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"
using namespace rapidxml;

namespace FVL {

	/**
	 * XML Writer class for FVL
	 *
	 * Used to generate output XML files
	 */
	class FVXMLWriter : public xml_document<> {

		private:
			string filename;
			ofstream out_stream;
			bool opened;

			xml_node<> *root;

		public:
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/

			/**
			 * Default constructor
			 *
			 * Initializes a Writer with no file specified. Open function should be called later to open a file
			 */
			FVXMLWriter();

			/**
			 * Default constructor
			 *
			 * Initializes a Writer for file specified in filename. If no file is specified, it is initialized to an empty string, and must later be give when saving the output
			 *
			 * \param filename The output file
			 */
			FVXMLWriter(string &file);
			FVXMLWriter(string &file, int x);

			/**
			 * Binds a file to the Writer
			 *
			 * \param filename The output file
			 */
			void open(string &file);

			/**
			 * Saves current output data to XML file
			 *
			 * \param filename If a filename was already specified upon construction, it is not necessary now, although a differente one can be used
			 */
			void save(string filename = string());

			/**
			 * Closes this writer
			 */
			void close();


			/************************************************
			 * OUTPUT (methods will be added as needed)
			 ***********************************************/

			// intended for output frame of an animation
			template<class T> void append(CFVArray<T> &vec, double time=0.0, string name="noname");

		private:

			/**
			 * Performs initializations on the class to prepare XML generation
			 */
			void init();

			/**
			 * Auxiliary functions to allocate and append attributes
			 */
			template<class T> void add_attribute(xml_node<> *node, string name, T value);
	};

}

#include "FVL/templates/FVXMLWriter.hpp"

#endif // _H_FVXMLWRITER
