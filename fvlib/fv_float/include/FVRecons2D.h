// ------ FVRecons2D.h ------
// S. CLAIN 2011/11
#ifndef _FVRECONS2D
#define _FVRECONS2D
#include "FVMesh2D.h"
#include "FVStencil.h"
#include "FVCell2D.h"
#include "FVVect.h"
#include "FVDenseM.h"
#include "FVPoint2D.h"
#include "FVPoint3D.h"
#include "FVGaussPoint.h"
#include "FVLib_config.h"

class FVRecons2D
{
private:
FVPoint2D<fv_float> _ref_point;
FVVect<fv_float> *_Vertex2DVect,*_Edge2DVect,*_Cell2DVect;  
FVVect<fv_float> *_coef,*_M;
FVStencil * _ptr_s;
FVDenseM<fv_float> *_A,*_Q;
fv_float _evalMean(void *ptr,size_t type,size_t alpha1, size_t alpha2); 
fv_float _ref_val;
size_t _degree,_Ncoef;
public:
// Constructors and destructors
FVRecons2D(){_ptr_s=NULL;_Vertex2DVect=NULL;_Edge2DVect=NULL;_Cell2DVect=NULL;
             _A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
             _ref_point=0.; _ref_val=0;_degree=0;_Ncoef=0;}
FVRecons2D(FVStencil *ptr_s)
    {
    _ptr_s=ptr_s;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0;_degree=0;_Ncoef=0;
    if(ptr_s->getReferenceType()==FVVERTEX2D) _ref_point=((FVVertex2D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE2D)   _ref_point=((FVEdge2D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVCELL2D)   _ref_point=((FVCell2D *)(_ptr_s->getReferenceGeometry()))->centroid;
    }
FVRecons2D(FVStencil *ptr_s, size_t degree)
    {
    _degree=degree;    
    _Ncoef=((_degree+2)*(_degree+1))/2-1;
    _ptr_s=ptr_s;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0;
    if(ptr_s->getReferenceType()==FVVERTEX2D) _ref_point=((FVVertex2D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE2D)   _ref_point=((FVEdge2D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVCELL2D)   _ref_point=((FVCell2D *)(_ptr_s->getReferenceGeometry()))->centroid;
    }    
~FVRecons2D(){if(_A) delete(_A); if(_Q) delete(_Q);if(_coef) delete(_coef);if(_M) delete(_M); }     
FVRecons2D(const FVRecons2D & rec); // copy constructor



// setStencil 
void setStencil(FVStencil &st){ FVRecons2D::setStencil(&st); }      
void setStencil(FVStencil *ptr_s)
    {
    _ptr_s=ptr_s;
    _ref_point=0;_degree=0;_Ncoef=0;
    if(ptr_s->getReferenceType()==FVVERTEX2D) _ref_point=((FVVertex2D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE2D)   _ref_point=((FVEdge2D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVCELL2D)   _ref_point=((FVCell2D *)(_ptr_s->getReferenceGeometry()))->centroid;       
     }
void setStencil(FVStencil &st, size_t degree){ FVRecons2D::setStencil(&st,degree); }      
void setStencil(FVStencil *ptr_s, size_t degree)
    {
    _ptr_s=ptr_s;
    _ref_point=0;   
    _degree=degree; 
    _Ncoef=((_degree+2)*(_degree+1))/2-1;
    if(ptr_s->getReferenceType()==FVVERTEX2D) _ref_point=((FVVertex2D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVEDGE2D)   _ref_point=((FVEdge2D *)(_ptr_s->getReferenceGeometry()))->centroid;    
    if(ptr_s->getReferenceType()==FVCELL2D)   _ref_point=((FVCell2D *)(_ptr_s->getReferenceGeometry()))->centroid;       
     }
// others     
void setPolynomialDegree(size_t degree){_degree=degree;_Ncoef=((_degree+2)*(_degree+1))/2-1;}    
size_t getPolynomialDegree(){return(_degree); } 
void setReferencePoint(FVPoint2D<fv_float> P){_ref_point=P;}
void setVectorVertex2D( FVVect<fv_float> & u){_Vertex2DVect=&u;}
void setVectorEdge2D( FVVect<fv_float> & u)  {_Edge2DVect=&u;}
void setVectorCell2D( FVVect<fv_float> & u)  {_Cell2DVect=&u;}
void doConservativeMatrix();
void computeConservativeCoef(); 
void doMatrix();
void computeCoef();
fv_float getValue(FVPoint2D<fv_float> P, size_t degree );    
fv_float getValue(FVPoint2D<fv_float> P){return(FVRecons2D::getValue(P,_degree));}    
FVPoint2D<fv_float> getDerivative(FVPoint2D<fv_float> P, size_t degree);
FVPoint2D<fv_float> getDerivative(FVPoint2D<fv_float> P){return(FVRecons2D::getDerivative(P,_degree));}
void clean()
    {
    if(_A) delete(_A); if(_Q) delete(_Q);if(_coef) delete(_coef);if(_M) delete(_M);
    _ptr_s=NULL;_Vertex2DVect=NULL;_Edge2DVect=NULL;_Cell2DVect=NULL;
    _A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0.; _ref_val=0;_degree=0;_Ncoef=0;
    }
    
};

FVPoint2D<size_t> alpha2D(size_t k);






#endif // define _FVRECONS2D
