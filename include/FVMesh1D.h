// ------ FVMesh1D.h ------
// for static mesh, we use the vector class 
// S. CLAIN 2011/07
#ifndef _FVMESH1D
#define _FVMESH1D
#include <string>
#include <vector>
#include "FVVertex1D.h"
#include "FVCell1D.h"
//
#include "XML.h"
using namespace std;

//class FVVertex1D;
//class FVCell1D;
class Gmsh;
class FVMesh1D
{
public:
     FVMesh1D();
     FVMesh1D(const char *);
     size_t read(const char *);
     size_t write(const char *);
     size_t getNbVertex(){ return _nb_vertex;}
     size_t getNbBoundaryVertex(){return _nb_boundary_vertex;}
     size_t getNbCell() {return _nb_cell;}
     FVVertex1D*  getVertex(size_t i){return &(_vertex[i]);} 
     FVCell1D*    getCell(size_t i){return &(_cell[i]);}     
     string getName(){ return _name;}
     void setName(const char * name){_name=name;}     
     FVVertex1D* beginVertex(){pos_v=0;if(pos_v<_nb_vertex) return&(_vertex[0]);else return(NULL);}
     FVVertex1D* nextVertex(){if(pos_v<_nb_vertex) return&(_vertex[pos_v++]);else return(NULL);}
     FVCell1D* beginCell(){pos_c=0;if(pos_c<_nb_cell) return&(_cell[0]);else return(NULL);};
     FVCell1D* nextCell(){if(pos_c<_nb_cell) return&(_cell[pos_c++]);else return(NULL);} 
     FVVertex1D* beginBoundaryVertex()
       {pos_bound_v=0;if(pos_bound_v<_nb_boundary_vertex) return(_boundary_vertex[0]);else return(NULL);}
     FVVertex1D* nextBoundaryVertex()
       {if(pos_bound_v<_nb_boundary_vertex) return(_boundary_vertex[pos_bound_v++]);else return(NULL);}  
     void Gmsh2FVMesh(Gmsh &); // convert a Gmsh struct into a FVMesh1D
private:
     void complete_data();
         
    vector<FVVertex1D> _vertex;
    vector<FVCell1D>  _cell;   
    vector<FVVertex1D *>  _boundary_vertex;
    size_t _nb_vertex,_nb_cell,_dim,_nb_boundary_vertex;
    size_t pos_v,pos_c,pos_bound_v;
    string _xml,_name;
    SparseXML _spxml;
};






#endif // define _FVMESH1D


