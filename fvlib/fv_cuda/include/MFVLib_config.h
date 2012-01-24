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

#define _DEBUG_	1

#if (_DEBUG_ == 0)
#define _DEBUG
#define _D
#else
#define _DEBUG	if (_DEBUG_)
#define _D(x)	(x)
#endif

#define _PROFILE_ 1

#if (_PROFILE_ == 0)
#define PROF_START(x)
#define PROF_STOP(x)
#define PROF_START_ONCE(i,x)
#define PROF_STOP_ONCE(i,x)
#else
#define PROF_START(x)			(x.start())
#define PROF_STOP(x)			(x.stop())
#define PROF_START_ONCE(i,x) 	(if (i == 0) PROF_START(x))
#define PROF_STOP_ONCE(i,x)		(if (i == 0) PROF_STOP(x))
#endif
