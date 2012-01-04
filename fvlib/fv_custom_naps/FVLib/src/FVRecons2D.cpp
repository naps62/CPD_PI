// ------ FVRecons2D.cpp ------
// S. CLAIN 2011/10
#include "FVRecons2D.h"



FVPoint2D<size_t> alpha2D(size_t k)
{
FVPoint2D<size_t> alpha;
size_t d=1;
if(k>1)  d+=1;
if(k>4)  d+=1;
if(k>8)  d+=1;
if(k>13) d+=1;
alpha.x=(((d+1)*(d+2))/2-(k+2));
alpha.y=d-alpha.x;
return(alpha);
}


FVRecons2D::FVRecons2D(const FVRecons2D & rec) // copy constructor
{
_ptr_s=rec._ptr_s;
_Vertex2DVect=rec._Vertex2DVect;
_Edge2DVect=rec._Edge2DVect;
_Cell2DVect=rec._Cell2DVect;
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
// compute mean value (x-c_x)^alpha1(y-c_y)^alpha2 on the geometrical entity
double FVRecons2D::_evalMean(void *ptr,size_t type,size_t alpha1,size_t alpha2)
{
FVPoint2D<double> P1,P2,P;
double sum,S,S_global,aux;  
sum=0.;
FVPoint2D<double> GPEdge;
FVPoint3D<double> GPCell;
FVGaussPoint1D G1D;
FVGaussPoint2D G2D;
// novas variáveis
FVCell2D *ptr_c;
FVEdge2D *ptr_e;
FVVertex2D *ptr_v1,*ptr_v2;
FVPoint2D<double> centroid;

switch(type)
   {
    case FVVERTEX2D:
    P=((FVVertex2D *)ptr)->coord;
    return(pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2));    
    break;
    case FVEDGE2D:
    P1=((FVEdge2D *)ptr)->firstVertex->coord;
    P2=((FVEdge2D *)ptr)->secondVertex->coord; 
    GPEdge=G1D.getPoint(5,1);
    P=GPEdge.x*P1+GPEdge.y*P2;
    sum+=G1D.getWeight(5,1)*pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2);
    GPEdge=G1D.getPoint(5,2);
    P=GPEdge.x*P1+GPEdge.y*P2;
    sum+=G1D.getWeight(5,2)*pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2);
    GPEdge=G1D.getPoint(5,3);
    P=GPEdge.x*P1+GPEdge.y*P2;
    sum+=G1D.getWeight(5,3)*pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2);    
    return(sum); 
    break; 
    //
    case FVCELL2D:
    ptr_c=((FVCell2D *)ptr);
    ptr_c->beginEdge();
    centroid=ptr_c->centroid; 
    S_global=0;
    while((ptr_e=ptr_c->nextEdge()))
        {
        ptr_v1=ptr_e->firstVertex;
        ptr_v2=ptr_e->secondVertex;
        S=Det(ptr_v1->coord-centroid,ptr_v2->coord-centroid)*0.5;
        if(S<0) S*=-1.;
        aux=0;
        for (size_t i=1;i<=G2D.getNbPoint(5);i++)
             { 
               GPCell=G2D.getPoint(5,i);
               P=GPCell.x*ptr_v1->coord+GPCell.y*ptr_v2->coord+GPCell.z*centroid;
               aux+=G2D.getWeight(5,i)*pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2);
             }    
         sum+=aux*S;
         S_global+=S;
         }
    return(sum/S_global);     
    break;
    default:
    cout<<"WARNING: unknown geometrical entity in FVReconstruction2D, found"<<type<<endl;    
    return(0); 
    break;    
   }
return(0);  
}
// Matrix associated to reconstruction with the conservative reference value 
void FVRecons2D::doConservativeMatrix()
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
FVPoint2D<size_t> al;
size_t alpha1,alpha2;
for(size_t j=0;j<_Ncoef;j++)
    { 
     al=alpha2D(j);   
     alpha1=al.x;alpha2=al.y;  
     (*_M)[j]=FVRecons2D::_evalMean(_ptr_s->getReferenceGeometry(),_ptr_s->getReferenceType(),alpha1,alpha2);
     _ptr_s->beginGeometry();
     while((ptr=_ptr_s->nextGeometry()))
          {
          size_t i=_ptr_s->getIndex();
          _A->setValue(i,j,FVRecons2D::_evalMean(ptr,_ptr_s->getType(),alpha1,alpha2)-(*_M)[j]);
          }
    }
_A->QRFactorize(*_Q);   
}
// Matrix associated to reconstruction without the conservative reference value 
void FVRecons2D::doMatrix()
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
FVPoint2D<size_t> al;
size_t alpha1,alpha2;
_ptr_s->beginGeometry();
while((ptr=_ptr_s->nextGeometry()))
      {
          _A->setValue(_ptr_s->getIndex(),0,1.);
      }
for(size_t j=0;j<_Ncoef;j++)
    {  
     al=alpha2D(j);   
     alpha1=al.x;alpha2=al.y;  
     (*_M)[j]=0;
     _ptr_s->beginGeometry();
     while((ptr=_ptr_s->nextGeometry()))
          {
          _A->setValue(_ptr_s->getIndex(),j+1,FVRecons2D::_evalMean(ptr,_ptr_s->getType(),alpha1,alpha2));
          }
    }
_A->QRFactorize(*_Q);  
}


// Polynomial coeffient  with the conservative reference value 
void FVRecons2D::computeConservativeCoef()
{
FVVect<double> B(_ptr_s->getNbGeometry()),X(_ptr_s->getNbGeometry()); 
void *ptr;
double  geo_val=0;
_ref_val=0;
size_t k;
switch(_ptr_s->getReferenceType())
  {
    case FVVERTEX2D:
    #ifdef _DEBUGS
    if(!_Vertex2DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex2DVect is empty"<<endl;
         break;
         }
    #endif
    k= ( (FVVertex2D *) _ptr_s->getReferenceGeometry())->label-1; 
    _ref_val=(*_Vertex2DVect)[k];
    break;
    case FVEDGE2D:
    #ifdef _DEBUGS
    if(!_Edge2DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Edge2DVect is empty"<<endl;
         break;
         }
    #endif
    k=( (FVEdge2D *) _ptr_s->getReferenceGeometry())->label-1;     
    _ref_val=(*_Edge2DVect)[k];    
    break; 
    case FVCELL2D:
    #ifdef _DEBUGS
    if(!_Cell2DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell2DVect is empty"<<endl;
         break;
         }
    #endif        
    k=  ((FVCell2D *) _ptr_s->getReferenceGeometry())->label-1; 
    _ref_val=(*_Cell2DVect)[k];
  }
_ptr_s->beginGeometry();  
while((ptr=_ptr_s->nextGeometry()))
    {
    switch(_ptr_s->getType())
       {
        case FVVERTEX2D:
        #ifdef _DEBUGS
        if(!_Vertex2DVect)
             {
             cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex2DVect is empty"<<endl;
             break;
             }
       #endif   
       k= ((FVVertex2D *) ptr)->label-1; 
       geo_val=(*_Vertex2DVect)[k];
       break;
       case FVEDGE2D:       
       #ifdef _DEBUGS
       if(!_Edge2DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Edge2DVect is empty"<<endl;
           break;
           }
       #endif 
       k=((FVEdge2D *) ptr)->label-1;
       geo_val=(*_Edge2DVect)[k];    
       break; 
       case FVCELL2D:         
       #ifdef _DEBUGS           
       if(!_Cell2DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell2DVect is empty"<<endl;
           break;
           }
       #endif    
       k=  ((FVCell2D *) ptr)->label-1;  
       geo_val=(*_Cell2DVect)[k];
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
void FVRecons2D::computeCoef()
{
FVVect<double> B(_ptr_s->getNbGeometry()),X(_ptr_s->getNbGeometry()); 
void *ptr;
double  geo_val=0;
size_t k;

_ptr_s->beginGeometry();  
while((ptr=_ptr_s->nextGeometry()))
    {
    switch(_ptr_s->getType())
       {
        case FVVERTEX2D:
        #ifdef _DEBUGS
        if(!_Vertex2DVect)
             {
             cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex2DVect is empty"<<endl;
             break;
             }
       #endif   
       k= ((FVVertex2D *) ptr)->label-1; 
       geo_val=(*_Vertex2DVect)[k];
       break;
       case FVEDGE2D:       
       #ifdef _DEBUGS
       if(!_Edge2DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Edge2DVect is empty"<<endl;
           break;
           }
       #endif 
       k=((FVEdge2D *) ptr)->label-1;
       geo_val=(*_Edge2DVect)[k];    
       break; 
       case FVCELL2D:         
       #ifdef _DEBUGS           
       if(!_Cell2DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell2DVect is empty"<<endl;
           break;
           }
       #endif    
       k=  ((FVCell2D *) ptr)->label-1;  
       geo_val=(*_Cell2DVect)[k];
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



double FVRecons2D::getValue(FVPoint2D<double> P,size_t d)
{
// basic method to replace with horner method
double val=_ref_val;
size_t k;
FVPoint2D<size_t> al; 
size_t alpha1,alpha2;
//cout<<"reference value="<<_ref_val<<endl;
for(k=0;k<_Ncoef;k++)
    {
    al=alpha2D(k);   
    alpha1=al.x;alpha2=al.y;
    //cout<<"coef["<<alpha1<<","<<alpha2<<"]="<<(*_coef)[k]<<" com M="<<(*_M)[k]<<endl;
    val+=(*_coef)[k]*(pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2)-(*_M)[k]);
    }
return(val);
}
// compute the gradient
FVPoint2D<double> FVRecons2D::getDerivative(FVPoint2D<double> P, size_t degree) 
{
// basic method to replace with horner method
FVPoint2D<double> val=0.;
size_t k;
FVPoint2D<size_t> al; 
size_t alpha1,alpha2;
for(k=0;k<_Ncoef;k++)
    {
    al=alpha2D(k);   
    alpha1=al.x;alpha2=al.y;
    if(alpha1>0)
        {
        //cout<<"power["<<alpha1<<","<<alpha2<<"]="<<alpha1*(*_coef)[k]<<" com M="<<(*_M)[k]<<endl;  
        val.x+=alpha1*(*_coef)[k]*pow(P.x-_ref_point.x,alpha1-1)*pow(P.y-_ref_point.y,alpha2);
        }
    if(alpha2>0)        
        {
        //cout<<"power["<<alpha1<<","<<alpha2<<"]="<<alpha2*(*_coef)[k]<<" com M="<<(*_M)[k]<<endl;  
        val.y+=alpha2*(*_coef)[k]*pow(P.x-_ref_point.x,alpha1)*pow(P.y-_ref_point.y,alpha2-1);
        }
    }
return(val);
}



