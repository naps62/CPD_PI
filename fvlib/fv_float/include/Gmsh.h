 
// ------ Gmsh.h ------
// S. CLAIN 2011/08
#ifndef _GMSH
#define _GMSH

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include<cstdlib>
//#include "XML.h"
#include "FVVect.h"
#include "FVio.h"
#include "FVMesh1D.h"
#include "FVMesh2D.h"
#include "FVMesh3D.h"
#include "FVPoint1D.h"
#include "FVPoint2D.h"
#include "FVPoint3D.h"
#include "FVLib_config.h"



//class FVMesh1D;
//class FVMesh2D;
//class FVMesh3D;
class GMElement
{
public:
    GMElement(){label=0;}
    
size_t label,type_element,code_physical, code_elementary,nb_node,dim;
size_t node[GMSH_NB_NODE_PER_ELEMENT];
void show()
     {cout<<"--- GMElement: label="<<label<<", type element="<<type_element<<", physical code="<<code_physical
          <<", dim="<<dim<<endl;
      for(size_t i=0;i<nb_node;i++) cout<<"node number="<<node[i]<<endl;
     }
};


class Gmsh
{
public:
     Gmsh();
     ~Gmsh();
     Gmsh(const char *);     // constructor to read a gmsh file
     
     size_t getNbNode(){return _nb_node;}
     size_t getNbElement(){return _nb_element;}     
     size_t getDim(){return _dim;}            
          
     void readMesh(const char *); // read a gmsh file (format .msh)
     void writeMesh(const char *); // write a gmsh file (format .msh) 
     void close();
     
     void FVMesh2Gmsh( FVMesh1D &); // convert a FVMesh1D struct into a Gmsh
     void FVMesh2Gmsh( FVMesh2D &); // convert a FVMesh2D struct into a Gmsh
     void FVMesh2Gmsh( FVMesh3D &);  // convert a FVMesh3D struct into a  Gmsh 

     FVVertex3D* getNode(const size_t i){return &(_node[i]);}
     GMElement* getElement(const size_t i){return &(_element[i]);}
   
     
     void writeVector( FVVect<fv_float> &, const size_t type,const char *name, fv_float time);    
     void writeVector(const FVVect<fv_float> &,const FVVect<fv_float> &,  const size_t type,const char *name, fv_float time);    
     void writeVector(const FVVect<fv_float> &,const FVVect<fv_float> &, const FVVect<fv_float> &, const size_t type,const char *name, fv_float time);      
     void writeVector( FVVect<FVPoint1D<fv_float> > &, const size_t type,const char *name, fv_float time);        
     void writeVector( FVVect<FVPoint2D<fv_float> > &, const size_t type,const char *name, fv_float time);       
     void writeVector( FVVect<FVPoint3D<fv_float> > &, const size_t type,const char *name, fv_float time);    

private:
    vector<FVVertex3D> _node;
    vector<GMElement> _element;
    size_t _nb_node,_nb_element,_dim,_nb_save;    
    string _name;
    bool _if_is_open,_of_is_open;
    ifstream  _if;
    ofstream _of;
};


#endif // end of _GMSH
