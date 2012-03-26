/**
 * \file FVParameters.h
 *
 * \author Miguel Palhas
 * \date 25-03-2012
 */

#ifndef _H_FVPARAMETERS
#define _H_FVPARAMETERS

#include "FVL/FVXMLReader.h"

namespace FVL {

	class FVParameters : public FVXMLReader {

		xml_node<> *param_list;

		public:
			FVParameters(const string filename);

			string		 getString(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			double 		 getDouble(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			int			 getInteger(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			unsigned int getUnsigned(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			bool		 getBoolean(const string param, const string key = FV_PARAM_DEFAULT_KEY);

	};
}

#endif
