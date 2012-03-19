/* ---------------------------------------------------------------------------
** Finite Volume Library
** 
** MFVLog.h
** Global Configs for FVL
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Test:	---
** -------------------------------------------------------------------------*/

#ifndef _H_FVGLOBAL
#define _H_FVGLOBAL

#include <climits>
#include <fstream>
#include <string>

/************************************************
 * UTILITIES
 ***********************************************/
#define NO_RIGHT_CELL	INT_MAX
#define NO_CELL			INT_MAX
#define MAX_EDGES_PER_CELL 4

/************************************************
 * STREAM MANIPULATION VALUES
 ***********************************************/
#define FV_PRECISION	12
#define FV_CHAMP		20

/************************************************
 * LOGGING OPTIONS
 ***********************************************/
#define FV_LOGMODE_APPEND	ofstream::app
#define FV_LOGMODE_WRITE 	0
#define FV_LOGMODE			FV_LOGMODE_APPEND

#define FV_LOGFILE string("FVLib.log")
#define FV_ERRFILE string("FVLib.err")
#define FV_PROFILE string("FVLib.prof")

/************************************************
 * DEBUG OPTIONS
 ***********************************************/
#define FV_DEBUG	1

#define _DEBUG	if (FV_DEBUG)

#if (FV_DEBUG == 0)
#define _D
#else
#define _D(x)	(x)
#endif

/************************************************
 * PROFILING OPTIONS
 ***********************************************/
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
