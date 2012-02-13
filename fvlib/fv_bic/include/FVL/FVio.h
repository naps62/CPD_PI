#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include "rapidxml/rapidxml.hpp"
using namespace rapidxml;
using namespace std;

namespace FVL {

	class FVio {
		public:
	
		// parse content of <filename> into xml_document given by <res>
		static void parse_xml(xml_document<> &res, vector<char> &vec, string &filename) {
	
			ifstream mesh_file(filename.c_str());
			string xml;
			string line;
	
			while(getline(mesh_file, line))
				xml += line;
	
			// make a safe-to-modify copy
			//vector<char> xml_copy(xml.begin(), xml.end());
			vec = vector<char>(xml.begin(), xml.end());
			//xml_copy.push_back('\0');
			vec.push_back('\0');
	
			res.parse<0>(&vec[0]);
		}
	
		template<class T>
		static bool str_cast(T &t, const string &s) {
			istringstream iss(s);
			return !(iss >> t).fail();
		}
	
	};
}
