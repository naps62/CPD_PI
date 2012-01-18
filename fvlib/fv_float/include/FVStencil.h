#ifndef __FVSTENCIL_H
#define __FVSTENCIL_H
#include "FVLib_config.h"
#include "FVVertex1D.h"
#include "FVCell1D.h"
#include "FVVertex2D.h"
#include "FVEdge2D.h"
#include "FVCell2D.h"
#include "FVVertex3D.h"
#include "FVEdge3D.h"
#include "FVFace3D.h"
#include "FVCell3D.h"
#include "FVLib_config.h"
#include <iostream>
#include <cstdio>
class FVStencil
{
private:
void*  _reference_geometry;
size_t _reference_type;
vector<void*>  *_geometry ; 
vector<size_t> *_type; 
size_t _nb_geometry,_pos;
public:

     FVStencil(){ 
                 _nb_geometry=0;_pos=0;_reference_geometry=NULL;_reference_type=NULL_ENTITY;_geometry=NULL;_type=NULL;
                 _geometry=new vector<void*> ;_type=new vector<size_t>;
                }
    ~FVStencil(){
                 if(_geometry) {delete(_geometry);_geometry=NULL;}
                 if(_type) {delete(_type);_type=NULL;}
                }  
     FVStencil(const FVStencil &st); // copy class
     void* beginGeometry(){_pos=0;if(_pos<_nb_geometry) return((*_geometry)[0]);else return(NULL);}
     void* nextGeometry(){if(_pos<_nb_geometry) return((*_geometry)[_pos++]);else return(NULL);}   
     void* getGeometry(size_t i){return ((*_geometry)[i]);}
     size_t getType(){if(_pos>0) return((*_type)[_pos-1]); else return(NULL_ENTITY);}
     size_t getIndex(){if(_pos>0) return (_pos-1); else return(0);}
     size_t getReferenceType(){return _reference_type;}     
     size_t getType(size_t i){return((*_type)[i]);}   
     void* getReferenceGeometry(){return (_reference_geometry);}         
     size_t getNbGeometry(){return(_nb_geometry);}
     void clean(){_nb_geometry=0;_reference_geometry=NULL;_reference_type=NULL_ENTITY;
                  _geometry->resize(0),_type->resize(0);}     
     void show();
     //
     void addStencil(FVVertex1D *ptr );     
     void setReferenceGeometry(FVVertex1D *ptr );
     void addStencil(FVVertex2D *ptr );     
     void setReferenceGeometry(FVVertex2D *ptr );
     void addStencil(FVVertex3D *ptr );     
     void setReferenceGeometry(FVVertex3D *ptr );     
     void addStencil(FVCell1D *ptr );     
     void setReferenceGeometry(FVCell1D *ptr );
     void addStencil(FVCell2D *ptr );     
     void setReferenceGeometry(FVCell2D *ptr );
     void addStencil(FVCell3D *ptr );     
     void setReferenceGeometry(FVCell3D *ptr );       
     void addStencil(FVEdge2D *ptr );     
     void setReferenceGeometry(FVEdge2D *ptr );
     void addStencil(FVEdge3D *ptr );     
     void setReferenceGeometry(FVEdge3D *ptr );      
     void addStencil(FVFace3D *ptr );     
     void setReferenceGeometry(FVFace3D *ptr ); 
private:

};

#endif // define __FVSTENCIL_H
