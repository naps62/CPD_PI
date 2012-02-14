/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLReader.h
** XML Reader class based on RapidXML
**
** Author: Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

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

namespace FVL {

	class FVXMLReader : public xml_document<> {
			
		private:
			vector<char> xml_data;

		public:
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			FVXMLReader(string filename);


			/************************************************
			 * METHODS
			 ***********************************************/
			template<class T>
			static bool str_cast(T &t, const string &s) {
				istringstream iss(s);
				return !(iss >> t).fail();
			}

		private:
			/************************************************
			 * PRIVATE METHODS
			 ***********************************************/
			string read_file(string filename);

	};
}

#endif // _H_FVXMLREADER
