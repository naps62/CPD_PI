// ------ FVMesh2D.h ------
// S. CLAIN 2011/07
#ifndef _FVMESH2D
#define _FVMESH2D
#include <string>
#include <vector>
#include "FVVertex2D.h"
#include "FVCell2D.h"
#include "FVEdge2D.h"
//
#include "FVLib_config.h"
#include "XML.h"
using namespace std;
class Gmsh;
class FVMesh2D
{
public:
     FVMesh2D();
     FVMesh2D(const char *);
     size_t read(const char *);
     size_t write(const char *);
     size_t getNbVertex(){return _nb_vertex;}
     size_t getNbCell() {return _nb_cell;}
     size_t getNbEdge() {return _nb_edge;}    
     size_t getNbBoundaryEdge(){return _nb_boundary_edge;}
     FVVertex2D*  getVertex(size_t i){return &(_vertex[i]);} 
     FVEdge2D*    getEdge(size_t i){return &(_edge[i]);}      
     FVCell2D*    getCell(size_t i){return &(_cell[i]);}    
     
     string getName(){ return _name;}
     void setName(const char * name){_name=name;}   
     FVVertex2D* beginVertex(){pos_v=0;if(pos_v<_nb_vertex) return&(_vertex[0]);else return(NULL);}
     FVVertex2D* nextVertex(){if(pos_v<_nb_vertex) return&(_vertex[pos_v++]);else return(NULL);}
     FVEdge2D* beginEdge(){pos_e=0;if(pos_e<_nb_edge) return&(_edge[0]);else return(NULL);}
     FVEdge2D* nextEdge(){if(pos_e<_nb_edge) return&(_edge[pos_e++]);else return(NULL);}    
     FVEdge2D* beginBoundaryEdge()
       {pos_bound_e=0;if(pos_bound_e<_nb_boundary_edge) return(_boundary_edge[0]);else return(NULL);}
     FVEdge2D* nextBoundaryEdge()
       {if(pos_bound_e<_nb_boundary_edge) return(_boundary_edge[pos_bound_e++]);else return(NULL);}   
       
     FVCell2D* beginCell(){pos_c=0;if(pos_c<_nb_cell) return&(_cell[0]);else return(NULL);}
     FVCell2D* nextCell(){if(pos_c<_nb_cell) return&(_cell[pos_c++]);else return(NULL);}    
     
     void Gmsh2FVMesh( Gmsh &); // convert a Gmsh struct into a FVMesh2D
private:
    void complete_data();
    vector<FVVertex2D> _vertex;
    vector<FVCell2D>  _cell;   
    vector<FVEdge2D>  _edge;  
    vector<FVEdge2D *>  _boundary_edge;  
    size_t _nb_vertex,_nb_cell,_nb_edge,_nb_boundary_edge,_dim;
    size_t pos_v,pos_c,pos_e,pos_bound_e;
    string _xml,_name;
    SparseXML _spxml;
};






#endif // define _FVMESH2D


