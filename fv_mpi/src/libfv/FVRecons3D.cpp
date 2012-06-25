// ------ FVRecons3D.cpp ------
// S. CLAIN 2011/12/21

#include "FVRecons3D.h"


FVPoint3D<size_t> alpha3D(size_t k1)//k1 start to 0
{
FVPoint3D<size_t> alpha;
size_t d1=1,d2=0,k2;

if(k1>2)  d1+=1;
if(k1>8)  d1+=1;
if(k1>18) d1+=1;
if(k1>33) d1+=1;
k2=k1+1-((d1)*(d1+1)*(d1+2))/6;

if(k2>0)  d2+=1;
if(k2>2)  d2+=1;
if(k2>5)  d2+=1;
if(k2>9)  d2+=1;
if(k2>14) d2+=1;
alpha.z=k2-((d2)*(d2+1))/2;
//cout<<"d1="<<d1<<", d2="<<d2<<", k2="<<k2<<endl;
alpha.y=d2-alpha.z;
alpha.x=d1-alpha.y-alpha.z;
return(alpha);
}

FVRecons3D::FVRecons3D(const FVRecons3D & rec) // copy constructor
{
_ptr_s=rec._ptr_s;
_Vertex3DVect=rec._Vertex3DVect;
_Edge3DVect=rec._Edge3DVect;
_Face3DVect=rec._Face3DVect;
_Cell3DVect=rec._Cell3DVect;
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

double FVRecons3D::_evalMean(void *ptr,size_t type,size_t alpha1,size_t alpha2,size_t alpha3)
{
FVPoint3D<double> P1,P2,P;
double sum,S,S_global,aux,V,V_global;
sum=0.;
FVPoint2D<double> GPEdge;
FVPoint3D<double> GPFace;
FVPoint4D<double> GPCell;
FVGaussPoint1D G1D;
FVGaussPoint2D G2D;
FVGaussPoint3D G3D;
FVCell3D *ptr_c;
FVFace3D *ptr_f;
FVEdge3D *ptr_e;
FVVertex3D *ptr_v1,*ptr_v2;
FVPoint3D<double> centroidF,centroidC;
switch(type)
   {
    case FVVERTEX3D:
    P=((FVVertex3D *)ptr)->coord;
    return(pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3));    
    //break;
    case FVEDGE3D:
    P1=((FVEdge3D *)ptr)->firstVertex->coord;
    P2=((FVEdge3D *)ptr)->secondVertex->coord; 
    GPEdge=G1D.getPoint(5,1);
    P=GPEdge.x*P1+GPEdge.y*P2;
    sum+=G1D.getWeight(5,1)*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3);
    GPEdge=G1D.getPoint(5,2);
    P=GPEdge.x*P1+GPEdge.y*P2;
    sum+=G1D.getWeight(5,2)*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3); 
    GPEdge=G1D.getPoint(5,3);
    P=GPEdge.x*P1+GPEdge.y*P2;
    sum+=G1D.getWeight(5,3)*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3);    
    return(sum);
    //break;   
    case FVFACE3D:
    ptr_f=((FVFace3D *)ptr);
    centroidF=ptr_f->centroid; 
    S_global=0;
    ptr_f->beginEdge();
    while((ptr_e=ptr_f->nextEdge()))
        {
        ptr_v1=ptr_e->firstVertex;
        ptr_v2=ptr_e->secondVertex;
        S= Norm(CrossProduct(ptr_v1->coord-centroidF,ptr_v2->coord-centroidF))*0.5;
        if(S<0) S*=-1.;
        aux=0;
        for (size_t i=1;i<=G2D.getNbPoint(5);i++)
             { 
               GPFace=G2D.getPoint(5,i);
               P=GPFace.x*ptr_v1->coord+GPFace.y*ptr_v2->coord+GPFace.z*centroidF;
               aux+=G2D.getWeight(5,i)*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3);
             }    
         sum+=aux*S;
         S_global+=S;
         }
    return(sum/S_global);  
    //break;
    case FVCELL3D:
    ptr_c=((FVCell3D *)ptr);
    centroidC=ptr_c->centroid; 
    V_global=0; 
    ptr_c->beginFace();    
    while((ptr_f=ptr_c->nextFace()))
        {
        centroidF=ptr_f->centroid;  
        ptr_f->beginEdge();
        while((ptr_e=ptr_f->nextEdge()))
            {
            ptr_v1=ptr_e->firstVertex;
            ptr_v2=ptr_e->secondVertex;
            V= Det(ptr_v1->coord-centroidC,ptr_v2->coord-centroidC,centroidF-centroidC)/6;
            if(V<0) V*=-1.;
            aux=0;
            for(size_t i=1;i<=G3D.getNbPoint(5);i++)
                { 
                 GPCell=G3D.getPoint(5,i);
                 P=GPCell.x*ptr_v1->coord+GPCell.y*ptr_v2->coord+GPCell.z*centroidF+GPCell.t*centroidC;
                 aux+=G3D.getWeight(5,i)*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3);
                }    
            sum+=aux*V;
            V_global+=V;
            }
        } 
    return(sum/V_global);    
    //break;    
    default:
    cout<<"WARNING: unknow geometrical entity in FVReconstruction3D"<<endl;    
    return(0); 
    //break;    
   }
//return(0);  
}
// Matrix associated to reconstruction without the conservative reference value 
void FVRecons3D::doConservativeMatrix()
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
FVPoint3D<size_t> al;
size_t alpha1,alpha2,alpha3;
for(size_t j=0;j<_Ncoef;j++)
    { 
     al=alpha3D(j);   
     alpha1=al.x;alpha2=al.y;alpha3=al.z;
     (*_M)[j]=FVRecons3D::_evalMean(_ptr_s->getReferenceGeometry(),_ptr_s->getReferenceType(),alpha1,alpha2,alpha3);
     _ptr_s->beginGeometry();
     while((ptr=_ptr_s->nextGeometry()))
          {
          size_t i=_ptr_s->getIndex();
          _A->setValue(i,j,FVRecons3D::_evalMean(ptr,_ptr_s->getType(),alpha1,alpha2,alpha3)-(*_M)[j]);
          }
    }
_A->QRFactorize(*_Q);   
}
// Matrix associated to reconstruction without the conservative reference value 
void FVRecons3D::doMatrix()
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
FVPoint3D<size_t> al;
size_t alpha1,alpha2,alpha3;
_ptr_s->beginGeometry();
while((ptr=_ptr_s->nextGeometry()))
      {
          _A->setValue(_ptr_s->getIndex(),0,1.);
      }
for(size_t j=0;j<_Ncoef;j++)
    {  
     al=alpha3D(j);   
     alpha1=al.x;alpha2=al.y;alpha3=al.z;
     (*_M)[j]=0;
     _ptr_s->beginGeometry();
     while((ptr=_ptr_s->nextGeometry()))
          {
          _A->setValue(_ptr_s->getIndex(),j+1,FVRecons3D::_evalMean(ptr,_ptr_s->getType(),alpha1,alpha2,alpha3));
          }
    }
_A->QRFactorize(*_Q);   
}

// Polynomial coeffient  with the conservative reference value 
void FVRecons3D::computeConservativeCoef()
{
    FVVect<double> B(_ptr_s->getNbGeometry()),X(_ptr_s->getNbGeometry()); 
void *ptr;
double  geo_val=0;
_ref_val=0;
size_t k;
switch(_ptr_s->getReferenceType())
  {
    case FVVERTEX3D:
    #ifdef _DEBUGS
    if(!_Vertex3DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex3DVect is empty"<<endl;
         break;
         }
    #endif
    k= ( (FVVertex3D *) _ptr_s->getReferenceGeometry())->label-1; 
    _ref_val=(*_Vertex3DVect)[k];
    break;
    case FVEDGE3D:
    #ifdef _DEBUGS
    if(!_Edge3DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Edge3DVect is empty"<<endl;
         break;
         }
    #endif
    k=( (FVEdge3D *) _ptr_s->getReferenceGeometry())->label-1;     
    _ref_val=(*_Edge3DVect)[k];    
    break; 
    case FVFACE3D:
    #ifdef _DEBUGS
    if(!_face3DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Face3DVect is empty"<<endl;
         break;
         }
    #endif        
    k=  ((FVFace3D *) _ptr_s->getReferenceGeometry())->label-1; 
    _ref_val=(*_Face3DVect)[k];
    break; 
    case FVCELL3D:
    #ifdef _DEBUGS
    if(!_Cell3DVect)
         {
         cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell3DVect is empty"<<endl;
         break;
         }
    #endif        
    k=  ((FVCell3D *) _ptr_s->getReferenceGeometry())->label-1; 
    _ref_val=(*_Cell3DVect)[k];    
  }
_ptr_s->beginGeometry();  
while((ptr=_ptr_s->nextGeometry()))
    {
    switch(_ptr_s->getType())
       {
        case FVVERTEX3D:
        #ifdef _DEBUGS
        if(!_Vertex3DVect)
             {
             cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex3DVect is empty"<<endl;
             break;
             }
       #endif   
       k= ((FVVertex3D *) ptr)->label-1; 
       geo_val=(*_Vertex3DVect)[k];
       break;
       case FVEDGE3D:       
       #ifdef _DEBUGS
       if(!_Edge3DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Edge3DVect is empty"<<endl;
           break;
           }
       #endif 
       k=((FVEdge3D *) ptr)->label-1;
       geo_val=(*_Edge3DVect)[k];    
       break; 
       case FVFACE3D:         
       #ifdef _DEBUGS           
       if(!_Face3DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Face3DVect is empty"<<endl;
           break;
           }
       #endif    
       k=  ((FVFace3D *) ptr)->label-1;  
       geo_val=(*_Face3DVect)[k];
       break; 
       case FVCELL3D:         
       #ifdef _DEBUGS           
       if(!_Cell3DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell3DVect is empty"<<endl;
           break;
           }
       #endif    
       k=  ((FVCell3D *) ptr)->label-1;  
       geo_val=(*_Cell3DVect)[k];       
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
void FVRecons3D::computeCoef()
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
        case FVVERTEX3D:
        #ifdef _DEBUGS
        if(!_Vertex3DVect)
             {
             cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Vertex3DVect is empty"<<endl;
             break;
             }
       #endif   
       k= ((FVVertex3D *) ptr)->label-1; 
       geo_val=(*_Vertex3DVect)[k];
       break;
       case FVEDGE3D:       
       #ifdef _DEBUGS
       if(!_Edge3DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Edge3DVect is empty"<<endl;
           break;
           }
       #endif 
       k=((FVEdge3D *) ptr)->label-1;
       geo_val=(*_Edge3DVect)[k];    
       break; 
       case FVFACE3D:         
       #ifdef _DEBUGS           
       if(!_Face3DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Face3DVect is empty"<<endl;
           break;
           }
       #endif    
       k=  ((FVFace3D *) ptr)->label-1;  
       geo_val=(*_Face3DVect)[k];
       break; 
       case FVCELL3D:         
       #ifdef _DEBUGS           
       if(!_Cell3DVect)
           {
           cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"vector _Cell3DVect is empty"<<endl;
           break;
           }
       #endif    
       k=  ((FVCell3D *) ptr)->label-1;  
       geo_val=(*_Cell3DVect)[k];       
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

double FVRecons3D::getValue(FVPoint3D<double> P,size_t d)
{
UNUSED(d);
// basic method to replace with horner method
double val=_ref_val;
size_t k;
FVPoint3D<size_t> al; 
size_t alpha1,alpha2,alpha3;
//cout<<"reference value="<<_ref_val<<endl;
for(k=0;k<_Ncoef;k++)
    {
    al=alpha3D(k);   
    alpha1=al.x;alpha2=al.y;alpha3=al.z;
    //cout<<"coef["<<alpha1<<","<<alpha2<<","<<alpha3<<"]="<<(*_coef)[k]<<" with M="<<(*_M)[k]<<endl;
    val+=(*_coef)[k]*(pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3)-(*_M)[k]);
    }
return(val);
}
// compute the gradient
FVPoint3D<double> FVRecons3D::getDerivative(FVPoint3D<double> P, size_t degree)
{
UNUSED(degree);
// basic method to replace with horner method
FVPoint3D<double> val=0.;
size_t k;
FVPoint3D<size_t> al; 
size_t alpha1,alpha2,alpha3;
for(k=0;k<_Ncoef;k++)
    {
    al=alpha3D(k);   
    alpha1=al.x;alpha2=al.y;alpha3=al.z;
    if(alpha1>0)
        {
        //cout<<"power["<<alpha1<<","<<alpha2<<","<<alpha3<<"]="<<alpha1*(*_coef)[k]<<" com M="<<(*_M)[k]<<endl;  
        val.x+=alpha1*(*_coef)[k]*pow(P.x-_ref_point.x,(double)(alpha1-1))*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)alpha3);
        }
    if(alpha2>0)        
        {
        //cout<<"power["<<alpha1<<","<<alpha2<<","<<alpha3<<"]="<<alpha2*(*_coef)[k]<<" com M="<<(*_M)[k]<<endl;  
        val.y+=alpha2*(*_coef)[k]*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)(alpha2-1))*pow(P.z-_ref_point.z,(double)alpha3);
        }
    if(alpha3>0)        
        {
        //cout<<"power["<<alpha1<<","<<alpha2<<","<<alpha3<<"]="<<alpha3*(*_coef)[k]<<" com M="<<(*_M)[k]<<endl;  
        val.z+=alpha3*(*_coef)[k]*pow(P.x-_ref_point.x,(double)alpha1)*pow(P.y-_ref_point.y,(double)alpha2)*pow(P.z-_ref_point.z,(double)(alpha3-1));
        }        

    }
return(val);
}


