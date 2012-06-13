#include <iostream>
using namespace std;
#include <tk/stopwatch.hpp>

namespace profile {
	tk::Stopwatch *s;
	//PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;

	long long time_pre;
	long long time_main_loop;
	long long time_pos;

	void init() {
		s = new tk::Stopwatch();

		time_pre       = 0;
		time_main_loop = 0;
		time_pos       = 0;
	}

	inline void output(std::ostream& out) {
		out << time_pre << ';' << time_main_loop << ';' << time_pos << endl;
	}

	void cleanup() {
		delete s;
	}

	inline void pre() {
		//cout << s->last();
		time_pre = s->last().microseconds();
	}

	inline void main_loop() {
		//cout << s->last();
		time_main_loop = s->last().microseconds();
	}

	inline void pos() {
		//cout << s->last();
		time_pos = s->last().microseconds();
	}
}

#define PROFILE_COUNTER              profile::s
#define PROFILE_INIT()               profile::init()
#define PROFILE_RETRIEVE_PRE()       profile::pre()
#define PROFILE_RETRIEVE_MAIN_LOOP() profile::main_loop()
#define PROFILE_RETRIEVE_POS()       profile::pos()
#define PROFILE_OUTPUT()             profile::output(cout)
#define PROFILE_CLEANUP()            profile::cleanup()
#define PROFILE_START() profile::s->start()
#define PROFILE_STOP()  profile::s->stop()

#define PROFILE_ZONES

#define OPTIM_LENGTH_AREA_RATIO
#define OPTIM_KERNELS

#include "../cuda.profile/main.cu"