// ------ FVRecons1D.h ------
// S. CLAIN 2011/10
#ifndef _FVRECONS1D
#define _FVRECONS1D
#include "FVMesh1D.h"
#include "FVStencil.h"
#include "FVVect.h"
#include "FVDenseM.h"
#include "FVPoint1D.h"
#include "FVGaussPoint.h"
#include "FVLib_config.h"

class FVRecons1D
{
private:
FVPoint1D<double> _ref_point;
FVVect<double> *_Vertex1DVect,*_Cell1DVect;  
FVVect<double> *_coef,*_M;
FVStencil * _ptr_s;
FVDenseM<double> *_A,*_Q;
double _evalMean(void *ptr,size_t type,size_t alpha); 
double _ref_val;
size_t _degree,_Ncoef;

public:
FVRecons1D(){_ptr_s=NULL;_Vertex1DVect=NULL;_Cell1DVect=NULL;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
             _ref_point=0.; _ref_val=0;_degree=0;_Ncoef=0;}
             
FVRecons1D(FVStencil *ptr_s)
    {
    _ptr_s=ptr_s;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0;_degree=0;_Ncoef=0;
    if(ptr_s->getReferenceType()==FVVERTEX1D) _ref_point=((FVVertex1D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVCELL1D) _ref_point=((FVCell1D *)(_ptr_s->getReferenceGeometry()))->centroid;
    }
FVRecons1D(FVStencil *ptr_s, size_t degree)
    {
    _degree=degree;    
    _Ncoef=_degree;
    _ptr_s=ptr_s;_A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0;
    if(ptr_s->getReferenceType()==FVVERTEX1D) _ref_point=((FVVertex1D *)(_ptr_s->getReferenceGeometry()))->coord;
    if(ptr_s->getReferenceType()==FVCELL1D) _ref_point=((FVCell1D *)(_ptr_s->getReferenceGeometry()))->centroid;
    }  
    
~FVRecons1D(){if(_A) delete(_A); if(_Q) delete(_Q);if(_coef) delete(_coef);if(_M) delete(_M);}  
FVRecons1D(const FVRecons1D & rec); // copy constructor
  
   
// setStencil    
void setStencil(FVStencil &st){ FVRecons1D::setStencil(&st); }      
void setStencil(FVStencil *ptr_s)
    {
    _ptr_s=ptr_s;
    _ref_point=0;_degree=0;_Ncoef=0;
     if(_ptr_s->getReferenceType()==FVVERTEX1D) _ref_point=((FVVertex1D *)(_ptr_s->getReferenceGeometry()))->coord;
     if(_ptr_s->getReferenceType()==FVCELL1D)  _ref_point=((FVCell1D *)(_ptr_s->getReferenceGeometry()))->centroid;        
     }
void setStencil(FVStencil &st, size_t degree){ FVRecons1D::setStencil(&st,degree); }      
void setStencil(FVStencil *ptr_s, size_t degree)
    {
    _ptr_s=ptr_s;
    _ref_point=0;   
    _degree=degree; 
    _Ncoef=_degree;
     if(_ptr_s->getReferenceType()==FVVERTEX1D) _ref_point=((FVVertex1D *)(_ptr_s->getReferenceGeometry()))->coord;
     if(_ptr_s->getReferenceType()==FVCELL1D)  _ref_point=((FVCell1D *)(_ptr_s->getReferenceGeometry()))->centroid;  
     }
//others     
void setPolynomialDegree(size_t degree){_degree=degree;_Ncoef=_degree;}    
size_t getPolynomialDegree(){return(_degree); } 
void setReferencePoint(double x){_ref_point.x=x;}
void setReferencePoint(FVPoint1D<double> P){_ref_point=P;}
void setVectorVertex1D( FVVect<double> & u){_Vertex1DVect=&u;}
void setVectorCell1D( FVVect<double> & u){_Cell1DVect=&u;}
void doConservativeMatrix();
void computeConservativeCoef();
void doMatrix();
void computeCoef();
double getValue(FVPoint1D<double> P, size_t degree); 
double getValue(FVPoint1D<double> P){return(FVRecons1D::getValue(P,_degree));} 
FVPoint1D<double> getDerivative(FVPoint1D<double> P, size_t degree);
FVPoint1D<double> getDerivative(FVPoint1D<double> P){return(FVRecons1D::getDerivative(P,_degree));}


void clean()
    {
    if(_A) delete(_A); if(_Q) delete(_Q);if(_coef) delete(_coef);if(_M) delete(_M);
    _ptr_s=NULL;_Vertex1DVect=NULL;_Cell1DVect=NULL;
    _A=NULL;_Q=NULL;_coef=NULL;_M=NULL;
    _ref_point=0.; _ref_val=0;_degree=0;_Ncoef=0;
    }
};

FVPoint1D<size_t> alpha1D(size_t k);


#endif // define _FVRECONS1D