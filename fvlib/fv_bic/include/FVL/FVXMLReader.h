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
	 * \todo known issue with \n characters being stripped from stream, resulting in numbers between two lines beeing concatenated
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
			FVXMLReader(string filename);


			/************************************************
			 * METHODS
			 ***********************************************/
			template<class T> static bool str_cast(T &t, const string &s);

			// Reads next child node to a vector from the XML file
			template<class T> void getVec	  	(CFVArray<T> &vec,	  double &time, string &name);

			// Reads next child node to a vector of 2D points
			template<class T> void getPoints2D	(CFVPoints2D<T> &vec, double &time, string &name);

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
