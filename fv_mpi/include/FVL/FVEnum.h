/**
 * \file FVEnum.h
 *
 * Holds all enumerators used by FVL
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVENUM
#define _H_FVENUM

namespace FVL {

	/************************************************
	 * MESH-RELATED ENUMS
	 ***********************************************/

	/**
	 * Types of edges in a 2D Mesh
	 */
	enum FVEdge2D_Type {
		FV_EDGE			= 0,		///< regular internal edge
		FV_EDGE_DIRICHLET	= 1,	///< border edge where dirichlet condition is applied
		FV_EDGE_NEUMMAN	= 2,		///< border edge where neumman condition is applied
		FV_EDGE_FAKE = 3			///< fake regular edge. used as a patch to link an edge to a cell that is not physically connected (to simulate cycling flux)
	};

	/**
	 * Types of cells in a 2D Mesh
	 */
	enum FVCell2D_Type {
		FV_CELL = 10				///< regular cell
	};

	/************************************************
	 * IO-RELATED ENUMS
	 ***********************************************/

	/**
	 * Types of I/O
	 */
	enum FVio_Type {
		FV_READ,		///< Read only file
		FV_WRITE,		///< Write only file
		FV_READ_WRITE	///< Read and Write file (use with care)
	};

	enum FV_LogType {
		FV_LOG,		///< Log message
		FV_WARNING,	///< Warning message
		FV_ERROR		///< Error message
	};
}

#endif // _H_FVENUM
