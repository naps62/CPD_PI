/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** MFVLog.h
** Global Configs for FVL
**
** Author: Miguel Palhas, mpalhas@gmail.com
** -------------------------------------------------------------------------*/

#pragma once
#define _H_M_FVLIB_CONFIG

#include <climits>
//#define NO_RIGHT_EDGE std::numeric_limits<unsigned int>::max()
#define NO_RIGHT_EDGE	INT_MAX

#define DEF_LOGFILE string("FVlib.log")
#define DEF_ERRFILE string("FVLib.err")
#define DEF_PROFILE string("FVLib.prof")

#define PROFILE 1
#if (_PROFILE_ == 0)
#define PROF_START(x)
#define PROF_STOP(x)
#else
#define PROF_START(x)	(x.start())
#define PROF_STOP(x)	(x.stop())
#endif
