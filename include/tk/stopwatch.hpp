#ifndef ___STOPWATCH_HPP___
#define ___STOPWATCH_HPP___

#include <ctime>

namespace tk
{
	class Stopwatch
	{
		static const unsigned NANOS_PER_SEC;

		static timespec add(timespec const &time1, timespec const &time2);
		static timespec diff(timespec const &time1, timespec const &time2);
		static void reset(timespec &time);



		//	instance
		bool _running;
		timespec _begin;
		timespec _end;
		timespec _total;
		timespec _partial;



		public:
		//	constructors
		Stopwatch();

		//	actions
		void start();
		void stop();
		void reset();
		void finish();
		
		//	getters
		//		pure
		timespec total();
		timespec partial();

		//		worked
		double total_s();
		double total_ms();
		double total_us();
		unsigned long long int total_ns();
	};
}

#endif/*___STOPWATCH_HPP___*/
