/**
 * \file FVParameters.cpp
 *
 * \author Miguel Palhas
 * \date 25-03-2012
 */

#include "FVL/FVParameters.h"
#include "FVL/FVXMLReader.h"

namespace FVL {

	FVParameters::FVParameters(const string filename) : FVXMLReader(filename) {
		param_list = root->first_node("parameters", 0, false);
	}

	string FVParameters::getString(const string param, const string key) {
		xml_node<> *param_node = param_list->first_node(param.c_str(), 0, false);
		xml_attribute<> *param_attrib = param_node->first_attribute(key.c_str(), 0, false);
		return string(param_attrib->value());
	}

	double FVParameters::getDouble(const string param, const string key) {
		string str = getString(param, key);
		double res;
		FVXMLReader::str_cast<double>(res, str);
		return res;
	}

	int FVParameters::getInteger(const string param, const string key) {
		string str = getString(param, key);
		int res;
		FVXMLReader::str_cast<int>(res, str);
		return res;
	}

	unsigned int FVParameters::getUnsigned(const string param, const string key) {
		string str = getString(param, key);
		unsigned int res;
		FVXMLReader::str_cast<unsigned int>(res, str);
		return res;
	}

	bool FVParameters::getBoolean(const string param, const string key) {
		string str = getString(param, key);
		bool res;
		FVXMLReader::str_cast<bool>(res, str);
		return res;
	}
}
