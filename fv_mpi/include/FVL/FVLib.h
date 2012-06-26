/**
 * \file FVLib.h
 *
 * \brief Global header file for FVL.
 *
 * Use this to include all library headers
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 * \todo make sure everything is included here
 */

#ifndef _H_FVLIB
#define _H_FVLIB

#include "FVL/FVMacros.h"
#include "FVL/FVEnum.h"

#include "FVL/FVMesh2D_SOA.h"
#include "FVL/FVRecons2D_SOA.h"

#include "FVL/FVErr.h"
#include "FVL/FVLog.h"

#include "FVL/FVXMLReader.h"
#include "FVL/FVXMLWriter.h"
#include "FVL/FVParameters.h"

#ifndef NO_CUDA
#ifdef __CUDACC__
#include "FVL/CFVLib.h"
#endif
#endif

#endif // _H_FVLIB
