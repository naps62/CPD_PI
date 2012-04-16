/**
 * \file FVMacros.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVMACROS
#define _H_FVMACROS

#include <climits>
#include <fstream>
#include <string>

/************************************************
 * UTILITIES
 ***********************************************/

#define FV_PARAM_DEFAULT_KEY	"value"	///< Default key to look for in a parameter file

#define NO_RIGHT_CELL	INT_MAX	///< Indicate non-existance of a right cell
#define NO_CELL			INT_MAX	///< Indicates non-existance of a cell
/**
 * \todo move this to INI file
 */
#define MAX_EDGES_PER_CELL	4	///< Maximum number of edges allowed for each cell
#define MAX_VERTEX_PER_CELL	4	///< Maximum number of verex for each cell (equal to MAX_EDGES_PER_CELL)

/************************************************
 * STREAM MANIPULATION VALUES
 ***********************************************/

#define FV_PRECISION	12	///< Precision to be used when outputing to a stream
#define FV_CHAMP		20	///< \todo comment this

/************************************************
 * LOGGING OPTIONS
 ***********************************************/
#define FV_LOGMODE_APPEND	ofstream::app		///< Append mode for #FVLog
#define FV_LOGMODE_WRITE 	0					///< Overwrite mode for #FVLog
/**
 * \todo move this to INI file
 */
#define FV_LOGMODE			FV_LOGMODE_APPEND	///< Currently selected mode for #FVLog

/**
 * \todo move this to INI file
 */
#define FV_LOGFILE string("/var/log/FVL/log.log")	///< Default log file for #FVLog

/**
 * \todo move this to INI file
 */
#define FV_ERRFILE string("/var/log/FVL/err.log")	///< Default error file for #FVErr

/**
 * \todo move this to INI file
 */
#define FV_PROFILE string("/var/lib/prof.log")	///< Default profiling file for #FVProfile and #CFVProfile

/************************************************
 * DEBUG OPTIONS
 ***********************************************/

/**
 * \todo move this to INI file
 */
#define FV_DEBUG	1			///< Set to 0 to disable debugging

#define _DEBUG	if (FV_DEBUG)

#if (FV_DEBUG == 0)
#define _D
#else
#define _D(x)	(x)
#endif

/************************************************
 * PROFILING OPTIONS
 ***********************************************/

/**
 * \todo move this to INI file
 */
#define FV_PROF 1				///< Set to 0 to disable profiling

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

#endif // _H_FVMACROS
