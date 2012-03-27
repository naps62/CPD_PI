#pragma once
#ifndef ___EVENT_HPP___
#define ___EVENT_HPP___

#include <string>
using std::string;

namespace papi
{
	class Event
	{
		int    _code;
		string _name;
	public:
		//
		//    constructor
		//
		/// Initializes a new instance using the event's name.
		Event(const string name);


		
		//
		//    getters
		//

		/// Retrieves the identifier code for this event.
		/**
		 * \return The code of this event.
		 */
		int code() const;



		//
		//    setters
		//

		/// Changes this event's name.
		/** This event's code is also changed to value associated with the given name in the PAPI library. The function PAPI_event_name_to_code(char*,int*) is used to decode the event's name accordingly.
		 */
		void name(const string name);
	};


	struct NativeEvent
	: public Event
	{
	protected:
		NativeEvent(string name);
	};
}

#endif//___EVENT_HPP___
