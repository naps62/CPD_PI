// ------ FVRecons1D.cpp ------
// S. CLAIN 2011/11
#include "FVRecons1D.h"



FVPoint1D<size_t> alpha1D(size_t k)
{
return(k+1);
}

FVRecons1D::FVRecons1D(const FVRecons1D & rec) // copy constructor
{
_ptr_s=rec._ptr_s;
_Vertex1DVect=rec._Vertex1DVect;
_Cell1DVect=rec._Cell1DVect;
_ref_point=rec._ref_point; 
_ref_val=rec._ref_val;
_degree=rec._degree;
_Ncoef=rec._Ncoef; 
if(!rec._A) _A=NULL;
else
    {
    _A=new   FVDenseM<double>;   
    _A->resize(_ptr_s->getNbGeometry(),_Ncoef);
    (*_A)=(*rec._A);        
    }   
if(!rec._Q) _Q=NULL;
else
    {
    _Q=new   FVDenseM<double>;
    _Q->resize(_ptr_s->getNbGeometry());
    (*_Q)=(*rec._Q);   
    } 
if(!rec._coef) _coef=NULL;
else
    {
    _coef= new FVVect<double>;
    _coef->resize(_Ncoef);    
    (*_coef)=(*rec._coef); 
    } 
if(!rec._M) _M=NULL;
else
    {
    _M=new FVVect<double>;
    _M->resize(_Ncoef);   
    (*_M)=(*rec._M);   
    } 
}  
             
double FVRecons1D::_evalMean(void *ptr,size_t type,size_t alpha)
{
double x1,x2,x,sum;  
x1=x2=x=sum=0.;
FVPoint2D<double> GP;
FVGaussPoint1D GPCell;
switch(type)
   {
    case FVVERTEX1D:
    x1=((FVVertex1D *)ptr)->coord.x;
    return(pow(x1-_ref_point.x,(double)alpha));    
    break;
    case FVCELL1D:
    x1=((FVCell1D *)ptr)->firstVertex->coord.x;
    x2=((FVCell1D *)ptr)->secondVertex->coord.x; 
    GP=GPCell.getPoint(5,1);
    x=GP.x*x1+GP.y*x2;
    sum+=GPCell.getWeight(5,1)*pow(x-_ref_point.x,(double)alpha);
    GP=GPCell.getPoint(5,2);
    x=GP.x*x1+GP.y*x2;
    sum+=GPCell.getWeight(5,2)*pow(x-_ref_point.x,(double)alpha);
    GP=GPCell.getPoint(5,3);
    x=GP.x*x1+GP.y*x2;
    sum+=GPCell.getWeight(5,3)*pow(x-_ref_point.x,(double)alpha);
    return(sum);
    break;     
    default:
    cout<<"WARNING: unknow geometrical entity in FVReconstruction1D"<<endl;    
    return(0); 
    break;     
   }
return(0);  
}
// Matrix associated to reconstruction with the conservative reference value 
void FVRecons1D::doConservativeMatrix()
{
   void *ptr;
#ifdef _DEBUGS
if(_ptr_s->nb_geometry<Ncoef-1)
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"Not enough entities for the reconstruction"<<endl;
#endif

         // create the matrix
_A=new   FVDenseM<double>;
_Q=new   FVDenseM<double>;
_A->resize(_ptr_s->getNbGeometry(),_Ncoef);
_Q->resize(_ptr_s->getNbGeometry());
_M=new FVVect<double>;
_M->resize(_Ncoef); 
FVPoint1D<size_t> al;
size_t alpha1;

for(size_t j=0;j<_Ncoef;j++)
    { 
     al=alpha1D(j);   
     alpha1=al.x;  
     (*_M)[j]=FVRecons1D::_evalMean(_ptr_s->getReferenceGeometry(),_ptr_s->getReferenceType(),alpha1);
     _ptr_s->beginGeometry();
     while((ptr=_ptr_s->nextGeometry()))
          {
          size_t i=_ptr_s->getIndex();
          _A->setValue(i,j,FVRecons1D::_evalMean(ptr,_ptr_s->getType(),alpha1)-(*_M)[j]);
          }
    }
_A->QRFactorize(*_Q);   
}
// Matrix associated to reconstruction without the conservative reference value 
void FVRecons1D::doMatrix()
{
   void *ptr;
#ifdef _DEBUGS
if(_ptr_s->nb_geometry<Ncoef)
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"Not enough entities for the reconstruction"<<endl;
#endif

         // create the matrix
_A=new   FVDenseM<double>;
_Q=new   FVDenseM<double>;
_A->resize(_ptr_s->getNbGeometry(),_Ncoef+1);
_Q->resize(_ptr_s->getNbGeometry());
_M=new FVVect<double>;
_M->resize(_Ncoef); 
FVPoint1D<size_t> al;
size_t alpha1;
_ptr_s->beginGeometry();
while((ptr=_ptr_s->nextGeometry()))
      {
          _A->setValue(_ptr_s->getIndex(),0,1.);
      }
for(size_t j=0;j<_Ncoef;j++)
    { 
     al=alpha1D(j);   
     alpha1=al.x;  
     (*_M)[j]=0;
     _ptr_s->beginGeometry();
     while((ptr=_ptr_s->nextGeometry()))
          {
          _A->setValue(_ptr_s->getIndex(),j+1,FVRecons1D::_evalMean(ptr,_ptr_s->getType(),alpha1));
          }
    }
_A->QRFactorize(*_Q);   
}
// Polynomial coeffient  with the conservative reference value 
void FVRecons1D::computeConservativeCoef()
{
FVVect<double> B(_ptr_s->getNbGeometry()),X(_ptr_s->getNbGeometry()); 
void *ptr;
double  geo_val=0;
_ref_val=0;
size_t k;
switch(_ptr_s->getReferenceType())
  {
    case FVVERTEX1D:
    k= ( (FVVertex1D *) _ptr_s->getReferenceGeometry())->label-1; 
    #ifdef _DEBUGS
    if(!_Vertex1DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex1DVect is empty"<<endl;
         break;
         }
    #endif
    _ref_val=(*_Vertex1DVect)[k];
    break;
    case FVCELL1D:
    #ifdef _DEBUGS
    if(!_Cell1DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell1DVect is empty"<<endl;
         break;
         }
    #endif        
    k=  ((FVCell1D *) _ptr_s->getReferenceGeometry())->label-1; 
    _ref_val=(*_Cell1DVect)[k];
  } 

_ptr_s->beginGeometry();
while((ptr=_ptr_s->nextGeometry()))
    {
    switch(_ptr_s->getType())
       {
        case FVVERTEX1D:
        #ifdef _DEBUGS
        if(!_Vertex1DVect)
             {
             cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex1DVect is empty"<<endl;
             break;
             }
       #endif      
       k= ((FVVertex1D *) ptr)->label-1; 
       geo_val=(*_Vertex1DVect)[k];
       break; 
       case FVCELL1D:  
       #ifdef _DEBUGS           
       if(!_Cell1DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell1DVect is empty"<<endl;
           break;
           }
       #endif 
       k=  ((FVCell1D *) ptr)->label-1;  
       geo_val=(*_Cell1DVect)[k];
       } 
  
    B[_ptr_s->getIndex()]=geo_val-_ref_val;
    }       
_Q->Mult(B, X);
_A->PartialBackwardSubstitution(X);
// create the vector
_coef= new FVVect<double>;
_coef->resize(_Ncoef);
for(size_t i=0;i<_Ncoef;i++) (*_coef)[i]=X[i];
}
// Polynomial coeffient  without the conservative reference value 
void FVRecons1D::computeCoef()
{
FVVect<double> B(_ptr_s->getNbGeometry()),X(_ptr_s->getNbGeometry()); 
void *ptr;
double  geo_val=0;
_ref_val=0;
size_t k;

_ptr_s->beginGeometry();
while((ptr=_ptr_s->nextGeometry()))
    {
    switch(_ptr_s->getType())
       {
        case FVVERTEX1D:
        #ifdef _DEBUGS
        if(!_Vertex1DVect)
             {
             cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex1DVect is empty"<<endl;
             break;
             }
       #endif      
       k= ((FVVertex1D *) ptr)->label-1; 
       geo_val=(*_Vertex1DVect)[k];
       break; 
       case FVCELL1D:  
       #ifdef _DEBUGS           
       if(!_Cell1DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell1DVect is empty"<<endl;
           break;
           }
       #endif 
       k=  ((FVCell1D *) ptr)->label-1;  
       geo_val=(*_Cell1DVect)[k];
       } 
  
    B[_ptr_s->getIndex()]=geo_val;
    }       
_Q->Mult(B, X);
_A->PartialBackwardSubstitution(X);
// create the vector
_coef= new FVVect<double>;
_coef->resize(_Ncoef);
for(size_t i=0;i<_Ncoef;i++) (*_coef)[i]=X[i+1];
_ref_val=X[0];
}





double FVRecons1D::getValue(FVPoint1D<double> P,size_t d)
{
UNUSED(d);
double val=_ref_val;
size_t k;
FVPoint1D<size_t> al;
size_t alpha1;
//cout<<"valeur de reference:"<<val<<endl;
for(k=0;k<_Ncoef;k++)
    {
    al=alpha1D(k);   
    alpha1=al.x;  
    val+=(*_coef)[k]*(pow(P.x-_ref_point.x,(double)alpha1)-(*_M)[k]);
    //cout<<"coef "<<k<<"="<<(*_coef)[k]<<" alapuissance "<<alpha1<<" et M="<<(*_M)[k]<<endl;
    }
return(val);
}
// comopute the derivative
FVPoint1D<double> FVRecons1D::getDerivative(FVPoint1D<double> P, size_t degree) 
{
UNUSED(degree);
FVPoint1D<double> val=0.;
size_t k;
FVPoint1D<size_t> al;
size_t alpha1;
for(k=0;k<_Ncoef;k++)
    {
    al=alpha1D(k);   
    alpha1=al.x;  
    val.x+=(*_coef)[k]*alpha1*pow(P.x-_ref_point.x,(double)(alpha1-1));
    //cout<<"coef "<<k<<"="<<alpha1*(*_coef)[k]<<" alapuissance "<<alpha1-1<<" et M="<<(*_M)[k]<<endl;
    }
return(val);
}








