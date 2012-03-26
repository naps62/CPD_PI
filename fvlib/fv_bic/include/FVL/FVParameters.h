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

	/**
	 * Parameter file reader class
	 *
	 * This class is used to parse a XML file following the specified format to input parameters to FVLib
	 *
	 * XML file structure should be as follows:
	 *
	 * <?xml version="1.0" encoding="UTF-8" ?>
	 * <fvlib>
	 *    <parameters>
	 *       <param1 value="param1_val" opt_arg="opt_arg_val" />
	 *       <param2 value="0.05" />
	 *    </parameters>
	 * </fvlib>
	 */
	class FVParameters : public FVXMLReader {

		xml_node<> *param_list;

		public:
			/**
			 * Constructor to parse a parameter file
			 *
			 * \param filename The XML file to parse
			 */
			FVParameters(const string filename);

			/**
			 * reads a string from the parameter list
			 *
			 * \param param Name of the tag to read
			 * \param key Propertie to read from the tag, defaults to "value"
			 * \return Value read as a string
			 */
			string		 getString(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			/**
			 * Reads a double from the parameter list
			 *
			 * \param param Name of the tag to read
			 * \param key Propertie to read from the tag, defaults to "value"
			 * \return Value read as a double
			 */
			double 		 getDouble(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			/**
			 * Reads an int from the parameter list
			 *
			 * \param param Name of the tag to read
			 * \param key Propertie to read from the tag, defaults to "value"
			 * \return Value read as an integer
			 */
			int			 getInteger(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			/**
			 * Reads an unsigned int from the parameter list
			 *
			 * \param param Name of the tag to read
			 * \param key Propertie to read from the tag, defaults to "value"
			 * \return Value read as an unsigned int
			 */
			unsigned int getUnsigned(const string param, const string key = FV_PARAM_DEFAULT_KEY);
			/**
			 * Reads a boolean from the parameter list
			 * The value should be stored as "0" or "1"
			 *
			 * \param param Name of the tag to read
			 * \param key Propertie to read from the tag, defaults to "value"
			 * \return Value read as a boolean
			 */
			bool		 getBoolean(const string param, const string key = FV_PARAM_DEFAULT_KEY);

	};
}

#endif
