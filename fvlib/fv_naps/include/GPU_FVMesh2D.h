// ------ FVMesh2D.h ------
// S. CLAIN 2011/07
#ifndef _GPU_FVMESH2D
#define _GPU_FVMESH2D
#include <string>
//#include <vector>
#include "FVVertex2D.h"
#include "FVCell2D.h"
#include "FVEdge2D.h"

#include "FVLib_config.h"
#include "FVMesh2D.h"
#include "XML.h"
#include "MFVLog.h"

using namespace std;
class Gmsh;

class GPU_FVMesh2D {
	private:
		FVLog logger;

	public:

		/**
		 * DualPtr holds two pointers
		 * cpu_ptr to handle data in cpu
		 * gpu_ptr, when not null, holds corresponding ptr to the gpu copied data
		 */
		typedef struct _s_DualPtr {
			double *cpu_ptr;
			double *gpu_ptr;
		} DualPtr;

		/**
		 * GPU_Point2D holds
		 */
		typedef struct _s_GPU_Point2D {
			DualPtr x;
			DualPtr y;
		} GPU_Point2D;

		/**
		 * EDGE INFO
		 */
		unsigned int num_edges;		// total number of edges
		GPU_Point2D edge_normals;	// normals for each edge
		DualPtr edge_lengths;		// length for each edge
		DualPtr edge_left_cells;	// size = num_edges. holds index of the cell left of each edge
		DualPtr edge_right_cells;	// size = num_edges. holds index of the cell right of each edge

		/**
		 * CELL INFO
		 */
		unsigned int num_cells;		// total number of cells
		DualPtr cell_areas;			// velocity of each cell

		/**
		 * OLD VARS
		 */
		//size_t _nb_vertex,_nb_cell,_nb_edge,_nb_boundary_edge,_dim;
		//size_t pos_v,pos_c,pos_e,pos_bound_e;
		//string _xml,_name;
		//SparseXML _spxml;

	public:

		/************************************************
		 * CONSTRUCTORS
		 ***********************************************/
		/* Not yet implemented */
		GPU_FVMesh2D();
		GPU_FVMesh2D(FVMesh2D &msh);
		~GPU_FVMesh2D();
		//GPU_FVMesh2D(const string &);
		//unsigned int read(const string &);
		//unsigned int write(const string &);
		

		/**
		 * Note: structs are public (for now), so the following methods are not necessary
		 * unsigned int getNbVertex()
		 * unsigned int getNbCell()
		 * unsigned int getNbEdge()
		 * unsigned int getNbBoundaryEdge()
		 * getVertex(unsigned int i)
		 * getEdge(unsigned int i)
		 * getCell(unsigned int i)
		 */

		//string getName() { return _name; }
		//void setName(const string &name) { _name = name; }

		/**
		 * Convert a Gmsh struct into a FVMesh2D
		 **/
		//void gmsh2FVMesh(Gmsh &);

	public:
	private:

		/**
		 * used by constructors
		 */
		void init();
		
		/************************************************
		 * IMPORT/EXPORT METHODS
		 ***********************************************/

		/**
		 * imports a default FVMesh2D format to GPU format 
		 */
		void import_FVMesh2D(FVMesh2D &);


		/************************************************
		 * MEMORY MANAGEMENT METHODS
		 ***********************************************/

		/**
		 * Aux method to set all pointers to NULL
		 * Used by empty constructor
		 */
		void zero_ptrs();
		/**
		 * Sets both pointers of a dual ptr to NULL
		 * Aux used by zero_ptrs
		 */
		void zero_dualPtr(DualPtr &ptr);

		/**
		 * Allocates CPU memory for data
		 */
		void alloc_cpu();

		/**
		 * aux function. allocs cpu_ptr of a given DualPtr, with a double[size] array
		 */
		void alloc_cpu(DualPtr &ptr, unsigned int size);

		/**
		 * deallocates all cpu memory
		 */
		void delete_cpu();

		/**
		 * deallocates memory for the cpu_ptr of a DualPtr
		 */
		void delete_cpu(DualPtr &ptr);

		/* Not yet implemented */
		//void complete_data();
};






#endif // define _GPU_FVMESH2D


