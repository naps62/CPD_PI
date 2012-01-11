// ------ FVRecons3D.h ------
// S. CLAIN 2011/10
#ifndef _FVRECONS3D
#define _FVRECONS3D
#include "FVMesh3D.h"
#include "FVStencil.h"
#include "FVCell3D.h"
#include "FVVect.h"
#include "FVDenseM.h"
#include "FVPoint3D.h"
#include "FVPoint4D.h"
#include "FVGaussPoint.h"
#include "FVLib_config.h"

class FVRecons3D
{
private:
FVPoint3D<double> _ref_point;
FVVect<double> *_Vertex3DVect,*_Edge3DVect,*_Face3DVect,*_Cell3DVect;    
FVVect<double> *_coef,*_M;
FVStencil * _ptr_s;
FVDenseM<double> *_A,*_Q;
double _evalMean(void *ptr,size_t type,size_t alpha1, size_t alpha2,size_t alpha3); 
double _ref_val;
size_t _degree,_Ncoef;

public:
FVRecons3D(){_ptr_s=NULL;_Vertex3DVect=NULL;_Edge3DVect=NULL;_Face3DVect=NULL;_Cell3DVect=NULL;_A=NULL;_Q=NULL;
              _coef=NULL;_M=NULL;_ref_point=0.; _ref_val=0;_degree=0;_Ncoef=0;}
FVRecons3D(FVStencil *ptr_s)
    {
    _ptr_s=ptr_s;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0;_degree=0;_Ncoef=0;
    if(ptr_s->getReferenceType()==FVVERTEX3D) _ref_point=((FVVertex3D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE3D)   _ref_point=((FVEdge3D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVFACE3D)   _ref_point=((FVFace3D *)(_ptr_s->getReferenceGeometry()))->centroid;
    if(ptr_s->getReferenceType()==FVCELL3D)   _ref_point=((FVCell3D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    }
FVRecons3D(FVStencil *ptr_s, size_t degree)
    {
    _degree=degree;    
    _Ncoef=((_degree+3)*(_degree+2)*(_degree+1))/6-1;
    _ptr_s=ptr_s;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0;
    if(ptr_s->getReferenceType()==FVVERTEX3D) _ref_point=((FVVertex3D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE3D)   _ref_point=((FVEdge3D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVFACE3D)   _ref_point=((FVFace3D *)(_ptr_s->getReferenceGeometry()))->centroid;
    if(ptr_s->getReferenceType()==FVCELL3D)   _ref_point=((FVCell3D *)(_ptr_s->getReferenceGeometry()))->centroid;   
    }      
~FVRecons3D(){if(_A) delete(_A); if(_Q) delete(_Q);if(_coef) delete(_coef);if(_M) delete(_M);}  
FVRecons3D(const FVRecons3D & rec); // copy constructor

// setStencil 
void setStencil(FVStencil &st){ FVRecons3D::setStencil(&st); }     
void setStencil(FVStencil *ptr_s)
    {
    _ptr_s=ptr_s;
    _ref_point=0;_degree=0;_Ncoef=0;
    if(ptr_s->getReferenceType()==FVVERTEX3D) _ref_point=((FVVertex3D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE3D)   _ref_point=((FVEdge3D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVFACE3D)   _ref_point=((FVFace3D *)(_ptr_s->getReferenceGeometry()))->centroid;
    if(ptr_s->getReferenceType()==FVCELL3D)   _ref_point=((FVCell3D *)(_ptr_s->getReferenceGeometry()))->centroid;          
     }
void setStencil(FVStencil &st, size_t degree){ FVRecons3D::setStencil(&st,degree); }      
void setStencil(FVStencil *ptr_s, size_t degree)
    {
    _ptr_s=ptr_s;
    _ref_point=0;   
    _degree=degree; 
    _Ncoef=((_degree+3)*(_degree+2)*(_degree+1))/6-1;
    if(ptr_s->getReferenceType()==FVVERTEX3D) _ref_point=((FVVertex3D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE3D)   _ref_point=((FVEdge3D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVFACE3D)   _ref_point=((FVFace3D *)(_ptr_s->getReferenceGeometry()))->centroid;
    if(ptr_s->getReferenceType()==FVCELL3D)   _ref_point=((FVCell3D *)(_ptr_s->getReferenceGeometry()))->centroid;       
     }
//other    
void setPolynomialDegree(size_t degree){_degree=degree;_Ncoef=((_degree+3)*(_degree)*(_degree+1))/6-1;}    
size_t getPolynomialDegree(){return(_degree); } 
void setReferencePoint(FVPoint3D<double> P){_ref_point=P;}
void setVectorVertex3D( FVVect<double> & u){_Vertex3DVect=&u;}
void setVectorEdge3D( FVVect<double> & u){_Edge3DVect=&u;}
void setVectorFace3D( FVVect<double> & u){_Face3DVect=&u;}
void setVectorCell3D( FVVect<double> & u){_Cell3DVect=&u;}
void doConservativeMatrix();
void computeConservativeCoef();
void doMatrix();
void computeCoef();
double getValue(FVPoint3D<double> P, size_t degree );    
double getValue(FVPoint3D<double> P){return(FVRecons3D::getValue(P,_degree));}    
FVPoint3D<double> getDerivative(FVPoint3D<double> P, size_t degree);
FVPoint3D<double> getDerivative(FVPoint3D<double> P){return(FVRecons3D::getDerivative(P,_degree));}
void clean()
    {
    if(_A) delete(_A); if(_Q) delete(_Q);if(_coef) delete(_coef);if(_M) delete(_M);
    _ptr_s=NULL;_Vertex3DVect=NULL;_Edge3DVect=NULL;_Face3DVect=NULL;_Cell3DVect=NULL;
    _A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0.; _ref_val=0;_degree=0;_Ncoef=0;
    }   
};

FVPoint3D<size_t> alpha3D(size_t k1);
#endif // define _FVRECONS3D




