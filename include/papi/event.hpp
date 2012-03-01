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
		Event(const string name);


		
		//
		//    getters
		//
		int code() const;



		//
		//    setters
		//
		void name(const string name);
	};
}

#endif//___EVENT_HPP___
