/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLReader.h
** XML Reader class based on RapidXML
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/
/**
 * \file CFVMesh2D.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVXMLREADER
#define _H_FVXMLREADER

#include <vector>
#include <sstream>
#include <string>
using std::string;
using std::vector;
using std::istringstream;

#include "rapidxml/rapidxml.hpp"
using namespace rapidxml;

#include "FVL/CFVArray.h"
#include "FVL/CFVPoints2D.h"

namespace FVL {

	/**
	 *
	 * \todo known issue with \\n characters being stripped from stream, resulting in numbers between two lines beeing concatenated if an additional space is not placed between them
	 */
	class FVXMLReader : public xml_document<> {
		
		private:
			vector<char> xml_data;
			xml_node<> *root;
			xml_node<> *current;

		public:
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			/**
			 * Constructor to create a XML Reader to parse a given file
			 *
			 * \param filename XML file to be parsed
			 */
			FVXMLReader(string filename);

			/************************************************
			 * GETTERS/SETTERS
			 ***********************************************/
			/**
			 * Returns node pointing to the root of the XML file
			 *
			 * \return Pointer to root node
			 */
			xml_node<>* getRootNode();

			/************************************************
			 * METHODS
			 ***********************************************/
			/**
			 * Helper function to cast a string to any given type
			 * Cast will be supported as long as stream operator is implemented for that type, allowing the operation (s >> t) to be performed
			 *
			 * \param t Reference to where the value (of type T) shall be stored
			 * \param s Reference to the string to parse
			 * \return True if the operation was successfull, false otherwise
			 */
			template<class T> static bool str_cast(T &t, const string &s);

			/**
			 * Reads next child node to a vector from the XML file
			 *
			 * \param vec Reference to the array where the values will be stored
			 * \param time Reference to where the timestamp associated with the XML vector will be stored
			 * \param name Reference to where the name of the vector will be stored
			 */
			template<class T> void getVec	  	(CFVArray<T> &vec,	  double &time, string &name);

			/**
			 * Reads next child node to a vector of 2-dimensional points from the XML file
			 *
			 * \param vec Reference to the array where the points will be stored
			 * \param time Reference to where the timestamp associated with the XML vector will be stored
			 * \param name Reference to where the name of the vector will be stored
			 */
			template<class T> void getPoints2D	(CFVPoints2D<T> &vec, double &time, string &name);

			/**
			 * Closes the parser, releasing all memory saved for it
			 */
			void close();

		private:
			/************************************************
			 * PRIVATE METHODS
			 ***********************************************/
			string read_file(string filename);

	};
}

#include "FVL/templates/FVXMLReader.hpp"

#endif // _H_FVXMLREADER
