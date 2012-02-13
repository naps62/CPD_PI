/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** MFVLog.h
** Global Configs for FVL
**
** Author: Miguel Palhas, mpalhas@gmail.com
** -------------------------------------------------------------------------*/

#ifndef _H_FVGLOBAL
#define _H_FVGLOBAL

#include <climits>
#include <fstream>
#include <string>

/**
 * Utilities
 */
//#define NO_RIGHT_EDGE std::numeric_limits<unsigned int>::max()
#define NO_RIGHT_EDGE	INT_MAX
#define NO_EDGE			INT_MAX
#define MAX_EDGES_PER_CELL 3

/**
 * Stream manipulation values
 */
#define FV_PRECISION	12
#define FV_CHAMP		20

/**
 * Logging options
 */
#define FV_LOGMODE_APPEND	ofstream::app
#define FV_LOGMODE_WRITE 	0
#define FV_LOGMODE			FV_LOGMODE_APPEND

#define FV_LOGFILE string("FVLib.log")
#define FV_ERRFILE string("FVLib.err")
#define FV_PROFILE string("FVLib.prof")

/**
 * Debug options
 */
#define FV_DEBUG	1

#define _DEBUG	if (FV_DEBUG)

#if (FV_DEBUG == 0)
#define _D
#else
#define _D(x)	(x)
#endif

/**
 * Profiling options
 */
#define FV_PROF 1

#if (FV_PROF == 0)
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

#endif // _H_FVGLOBAL
