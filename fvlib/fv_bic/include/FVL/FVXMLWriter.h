/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVXMLWriter.h
** XML Writer class based on RapidXML
**
** Author: Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** -------------------------------------------------------------------------*/

#ifndef _H_FVXMLWRITER
#define _H_FVXMLWRITER

#include <sstream>
#include <iomanip>
using namespace std;

#include "FVL/CFVVect.h"
#include "rapidxml/rapidxml.hpp"
using namespace rapidxml;

namespace FVL {
	class FVXMLWriter : public xml_document<> {

		private:
			xml_node<> *root;

		public:
			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			FVXMLWriter();

			void init();

			/************************************************
			 * OUTPUT (methods will be added as needed)
			 ***********************************************/

			template<class T>
			void add_attribute(xml_node<> *node, string name, T value) {
				stringstream ss;
				ss << value;
				string str_value(ss.str());
				node->append_attribute(
						this->allocate_attribute(
							this->allocate_string(name.c_str()),
							this->allocate_string(str_value.c_str())));
			}
			// intended for output frame of an animation
			template<class T>
			void append(CFVVect<T> &vec, double time=0.0, string name="noname") {
				stringstream ss;
				ss << scientific << setprecision(FV_PRECISION) << setw(FV_CHAMP);
				for(unsigned int i = 0; i < vec.size(); ++i) {
					ss << vec[i] << " ";
				}
				char* value_str = this->allocate_string(ss.str().c_str());
		
				xml_node<> *node = this->allocate_node(
						node_data,
						this->allocate_string("FIELD"),
						value_str);
		
				// append attributes
				unsigned int nbvec = 1;
				this->add_attribute<unsigned int>	(node, string("size"), vec.size());
				this->add_attribute<unsigned int>	(node, string("nbvec"), nbvec);
				this->add_attribute<double>			(node, string("time"), time);
				this->add_attribute<string>			(node, string("name"), name);

				root->append_node(node);
			}

		private:
			// aux functions to allocate and append attributes
	};
}

#endif // _H_FVXMLWRITER
