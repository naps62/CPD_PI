#include <fstream>
#include <string>
#include <set>
#include <map>

#include "FVL/FVRecons2D_SOA.h"
#include "FVLib_config.h"
#include "FVPoint2D.h"

#include "rapidxml/rapidxml.hpp"
#include "FVL/FVXMLReader.h"
#include "FVL/FVErr.h"

using namespace rapidxml;

namespace FVL {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	FVRecons2D_SOA::FVRecons2D_SOA(FVMesh2D_SOA &mesh) {
		alloc(mesh.num_cells, mesh.num_edges);
	}

	/************************************************
	 * MEMORY MANAGEMENT METHODS
	 ***********************************************/
	void FVRecons2D_SOA::alloc(unsigned int cells, unsigned int edges) {
		if (cells <= 0 || edges <= 0) {
			string msg = "num edges/cells not valid for allocation";
			FVErr::error(msg, -1);
		}

		u_ij 		= CFVArray<double>(edges);
		u_ji 		= CFVArray<double>(edges);
		F_ij 		= CFVArray<double>(edges);
		F_ij_old	= CFVArray<double>(edges);
		edge_state 	= CFVArray<bool>(edges);
		cell_state 	= CFVArray<bool>(cells);
		degree		= CFVArray<unsigned int>(cells);
	}
}

