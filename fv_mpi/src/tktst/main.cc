#include <tk/stopwatch.hpp>

#include <cstdio>
#include <iostream>

using std::endl;
using std::cout;
using tk::Stopwatch;
using tk::Time;

int main()
{
	Stopwatch program;
	do
	{
		program.toggle();
		Time total = program.total();
		cout
			<<	total	<<	endl
//			<<	total.nanoseconds()	<<	" ns"	<<	endl
			<<	total.microseconds()	<<	" us"	<<	endl
			<<	total.miliseconds()	<<	" ms"	<<	endl
			<<	total.seconds()	<<	" s"	<<	endl
			<<	total.minutes()	<<	" min"	<<	endl
			<<	total.hours()	<<	" h"	<<	endl
			;
	}
	while ( getchar() != 'q' );
	
	return 0;
}
