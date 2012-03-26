// ------ FVMesh3D.h ------
// S. CLAIN 2011/07
#ifndef _FVMESH3D
#define _FVMESH3D

#include <string>
#include <vector>
#include "FVVertex3D.h"
#include "FVEdge3D.h"
#include "FVFace3D.h"
#include "FVCell3D.h"
//
#include "FVLib_config.h"
#include "XML.h"
//using namespace std;
class Gmsh;

class FVMesh3D
{
public:
     FVMesh3D();
     FVMesh3D(const char *);
     size_t read(const char *);
     size_t write(const char *);
     size_t getNbVertex(){return _nb_vertex;}
     size_t getNbEdge() {return _nb_edge;}    
     size_t getNbFace() {return _nb_face;}  
     size_t getNbBoundaryFace(){return _nb_boundary_face;}
     size_t getNbCell() {return _nb_cell;}
 
     FVVertex3D*  getVertex(size_t i){return &(_vertex[i]);} 
     FVEdge3D*    getEdge(size_t i){return &(_edge[i]);}   
     FVFace3D*    getFace(size_t i){return &(_face[i]);}      
     FVCell3D*    getCell(size_t i){return &(_cell[i]);}   
     
  
     
     string getName(){ return _name;}
     void setName(const char * name){_name=name;}   
     FVVertex3D* beginVertex(){pos_v=0;if(pos_v<_nb_vertex) return&(_vertex[0]);else return(NULL);}
     FVVertex3D* nextVertex(){if(pos_v<_nb_vertex) return&(_vertex[pos_v++]);else return(NULL);}
     
     FVEdge3D* beginEdge(){pos_e=0;if(pos_e<_nb_edge) return&(_edge[0]);else return(NULL);}
     FVEdge3D* nextEdge(){if(pos_e<_nb_edge) return&(_edge[pos_e++]);else return(NULL);}   
     
     FVFace3D* beginFace(){pos_f=0;if(pos_f<_nb_face) return&(_face[0]);else return(NULL);}
     FVFace3D* nextFace(){if(pos_f<_nb_face) return&(_face[pos_f++]);else return(NULL);}   
     
     FVFace3D* beginBoundaryFace()
       {pos_bound_f=0;if(pos_bound_f<_nb_boundary_face) return(_boundary_face[0]);else return(NULL);}
     FVFace3D* nextBoundaryFace()
       {if(pos_bound_f<_nb_boundary_face) return(_boundary_face[pos_bound_f++]);else return(NULL);}      
     
     FVCell3D* beginCell(){pos_c=0;if(pos_c<_nb_cell) return&(_cell[0]);else return(NULL);}
     FVCell3D* nextCell(){if(pos_c<_nb_cell) return &(_cell[pos_c++]);else return(NULL);}    
     
     void Gmsh2FVMesh( Gmsh &);  // convert a Gmsh struct into a FVMesh3D    

private:
    void complete_data();
    vector<FVVertex3D> _vertex;
    vector<FVEdge3D>   _edge;     
    vector<FVFace3D>   _face;   
    vector<FVFace3D *>  _boundary_face;
    vector<FVCell3D>   _cell;   

    
    size_t _nb_vertex,_nb_cell,_nb_edge,_nb_face,_nb_boundary_face,_dim;
    size_t pos_v,pos_c,pos_e,pos_f,pos_bound_f;
    string _xml,_name;
    SparseXML _spxml;
};


#endif // define _FVMESH3D



