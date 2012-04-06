#ifndef PROFILE_START
#define PROFILE_START() PROFILE_COUNTER->start()
#endif

#ifndef PROFILE_STOP
#define PROFILE_STOP() PROFILE_COUNTER->stop()
#endif

#include "../polu.aos/main.cpp"
