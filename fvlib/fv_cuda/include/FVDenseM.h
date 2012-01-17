// ------ FVVDenseMatrice.h ------
// S. CLAIN 2011/07
#ifndef _FVDenseM
#define _FVDenseM
//-----------
#include "FVLib_config.h"
#include "FVGlobal.h"
#include "FVVect.h"
#include "FVPoint2D.h"
#include "FVPoint3D.h"
#include "FVPoint4D.h"
#include <valarray>
#include <iostream>
#include <cstdio>
#include <omp.h>

using std::cout;
using std::endl;
template<class T_> class FVDenseM : public valarray<T_>
{
protected:

public:
valarray<T_>  a;  // the place for the matrice
valarray<size_t> row_perm; // the row permutation index vector for the LU factorization
size_t nb_cols,nb_rows,length; // row and column


    FVDenseM();
    FVDenseM(size_t );
    FVDenseM(size_t , size_t );
    FVDenseM(const FVDenseM<T_> &m);
    size_t getNbColumns() {return nb_cols;}
    size_t getNbRows(){ return nb_rows;}
    size_t getLength(){ return length;}    
    valarray<T_> * getTab(){return &a;}
    void resize(size_t );
    void resize(size_t , size_t );
    void setValue(size_t i, size_t j, const T_ &val);
    void addValue(size_t i, size_t j, const T_ &val);    
    T_ getValue(size_t i, size_t j)  ;
    void setLine  (size_t i, const FVVect<T_> &line);
    void setColumn(size_t j, const FVVect<T_> &column);
    FVDenseM<T_> & operator=(const T_ &x);
    FVDenseM<T_> & operator+=(const FVDenseM<T_> &m);
    FVDenseM<T_> & operator-=(const FVDenseM<T_> &m);
    FVDenseM<T_> & operator/=(const T_ &val);
    FVDenseM<T_> & operator*=(const T_ &val);
    FVDenseM<T_> & operator+=(const T_ &val);
    FVDenseM<T_> & operator-=(const T_ &val);
    //FVVect<T_> getColumn(size_t j) const;
    //FVVect<T_> getRow(size_t i) const
    void Mult(const FVVect<T_> &, FVVect<T_> &) ;
    void TransMult(const FVVect<T_> &, FVVect<T_> &)   ;  
    // resolution method
    // Gauss method with pivot
    void Gauss(FVVect<fv_float> &) ;
    void Gauss(FVPoint2D<fv_float> &);
    void Gauss(FVPoint3D<fv_float> &);   
    void Gauss(FVPoint4D<fv_float> &);    
    //L U decomposition, we assume diag(L) are unit  
    void LUFactorize() ;   
    void ForwardSubstitution(FVVect<fv_float> &) ;// for lower matrix assuming identity on the diagonal 
    void BackwardSubstitution(FVVect<fv_float> &) ;// for upper matrix
    void ForwardSubstitution(FVPoint2D<fv_float> &) ;// for lower matrix assuming identity on the diagonal
    void BackwardSubstitution(FVPoint2D<fv_float> &) ;// for upper matrix
    void ForwardSubstitution(FVPoint3D<fv_float> &) ;// for lower matrix assuming identity on the diagonal
    void BackwardSubstitution(FVPoint3D<fv_float> &) ;// for upper matrix    
    void ForwardSubstitution(FVPoint4D<fv_float> &) ;// for lower matrix assuming identity on the diagonal
    void BackwardSubstitution(FVPoint4D<fv_float> &) ;// for upper matrix      
    //void decompositionLLtrans() const  ;    //LL^t decomposition,  
    //void decompositionLDLtrans() const  ; 
    void LUFactorizePivoting() ;   //L U decomposition, we assume diag(L) are unit    
    void ForwardSubstitutionPivoting(FVVect<fv_float> &) ;    
    void BackwardSubstitutionPivoting(FVVect<fv_float> &) ;    
    void ForwardSubstitutionPivoting(FVPoint2D<fv_float> &) ;    
    void BackwardSubstitutionPivoting(FVPoint2D<fv_float> &) ;   
    void ForwardSubstitutionPivoting(FVPoint3D<fv_float> &) ;     
    void BackwardSubstitutionPivoting(FVPoint3D<fv_float> &) ;    
    void ForwardSubstitutionPivoting(FVPoint4D<fv_float> &) ;         
    void BackwardSubstitutionPivoting(FVPoint4D<fv_float> &) ;    
    
    
    void QRFactorize(FVDenseM<fv_float> &)   ;  //QR decomposition of nrXnc matrix with nr>=nc overdetermine
    void PartialBackwardSubstitution( FVVect<fv_float> & ); // backward substitution
                                                              // with nr>=nc 
    void show();
    // produit matric matrice
    // produit matrice vecteur
};

///////////////////////////////////////////////////////////////////////////////
//                         I M P L E M E N T A T I O N                       //
///////////////////////////////////////////////////////////////////////////////

//construct an empty matrix
template<class T_>
FVDenseM<T_>::FVDenseM()
    {
    nb_rows = nb_cols = 0,length=0;
    }
// construct a square matrix
template<class T_>
FVDenseM<T_>::FVDenseM(size_t size)
    {
    nb_rows = nb_cols = size;
    length=nb_rows*nb_cols;
    a.resize(length);
    a = static_cast<T_>(0);
    row_perm.resize(nb_rows);
    }
// construct a rectangular matrix
template<class T_>
FVDenseM<T_>::FVDenseM(size_t nr, size_t nc)
    {
    nb_rows = nr;nb_cols = nc;
    length=nb_rows*nb_cols;
    a.resize(length);
    a = static_cast<T_>(0);
    row_perm.resize(nb_rows);
    }
// copy dense matrix    
template<class T_>
FVDenseM<T_>::FVDenseM(const FVDenseM<T_> &m)
{
   resize(m.nb_rows,m.nb_cols);
   nb_rows = m.nb_rows;nb_cols = m.nb_cols;
   length=nb_rows*nb_cols;
   a.resize(length);
   a = m.a;
   row_perm.resize(nb_rows);
   row_perm=m.row_perm;
}  
// initialize with a  constant
template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator=(const T_ &val)
{
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] = val;
   return *this;
}
//  define the size for a square matrix
template<class T_>
void FVDenseM<T_>::resize(size_t size)
    {
    nb_rows = nb_cols = size;
    length=nb_rows*nb_cols;
    a.resize(length);  
    row_perm.resize(nb_rows);    
    }
// define the size for a rectangular matrix
template<class T_>
void FVDenseM<T_>::resize(size_t nr, size_t nc)
    {
    nb_rows = nr;
    nb_cols = nc;
    length=nb_rows*nb_cols;
    a.resize(length)   ; 
    row_perm.resize(nb_rows);    
    }
   

// set a value  
template<class T_>
void FVDenseM<T_>::setValue(size_t i, size_t j, const T_ &val) 
    {
#ifdef _DEBUGS
    size_t err=(i<0)+2*(i>=nb_rows)+4*(j<0)+8*(j>=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
     a[nb_cols*i+j]=val;
    }
 // add a value  
template<class T_>
void FVDenseM<T_>::addValue(size_t i, size_t j, const T_ &val) 
    {
#ifdef _DEBUGS
    size_t err=(i<0)+2*(i>=nb_rows)+4*(j<0)+8*(j>=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
     a[nb_cols*i+j]+=val;
    }   
// return a value
template<class T_>
T_  FVDenseM<T_>::getValue(size_t i, size_t j)
    {
#ifdef _DEBUGS
    size_t err=(i<0)+2*(i>=nb_rows)+4*(j<0)+8*(j>=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
    return a[nb_cols*i+j];
    }  

// set a line 
template<class T_>
void FVDenseM<T_>::setLine(size_t i, const FVVect<T_> &line) 
    {
#ifdef _DEBUGS
    size_t err=(i<0)+2*(i>=nb_rows)+4*(line.size()>nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
    for(size_t j=0;j<nb_cols;j++)  a[nb_cols*i+j]=line[j];
    }    
// set a column 
template<class T_>
void FVDenseM<T_>::setColumn(size_t j, const FVVect<T_> &column) 
    {
#ifdef _DEBUGS
    size_t err=(j<0)+2*(j>=nb_cols)+4*(column.size()>nb_rows);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
    for(size_t i=0;i<nb_rows;j++)  a[nb_cols*i+j]=column[i];
    }        
    
    
/*-------- operators overload -------*/ 

template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator+=(const FVDenseM<T_> &m)
{
#ifdef _DEBUGS
    size_t err=(m.nb_rows!=nb_rows)+2*(m.nb_cols!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
//  #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] += m.a[i];
   return *this;
}


template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator-=(const FVDenseM<T_> &m)
{
  #ifdef _DEBUGS
    size_t err=(m.nb_rows!=nb_rows)+2*(m.nb_cols!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] -= m.a[i];
   return *this;
}







template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator+=(const T_ &val)
{
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] += val;
   return *this;
}


template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator-=(const T_ &val)
{
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] -= val;
   return *this;
}


template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator/=(const T_ &val)
{
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] /= val;
   return *this;
}

template<class T_>
FVDenseM<T_> & FVDenseM<T_>::operator*=(const T_ &val)
{
  //#pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] *= val;
   return *this;
}

template<class T_>
FVDenseM<T_> operator+(const FVDenseM<T_> &aa, const FVDenseM<T_> &bb)
{
#ifdef _DEBUGS
    size_t err=(aa.nb_rows!=bb.nb_rows())+2*(aa.nb_cols()!=bb.nb_cols());
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif  
    FVDenseM<T_> c(aa); 
 // #pragma omp parallel for num_threads(nb_thread)   
    for (size_t i=0; i<c.length; ++i)
                c.a[i] =aa.a[i]+ bb.a[i];
    return c;
}

template<class T_>
FVDenseM<T_> operator-(const FVDenseM<T_> &aa, const FVDenseM<T_> &bb)
{
#ifdef _DEBUGS
    size_t err=(aa.nb_rows!=bb.nb_rows)+2*(aa.nb_cols!=bb.nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif  
   FVDenseM<T_> c(aa);
  //#pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<c.length; ++i)
      c.a[i] =aa.a[i]- bb.a[i];
   return c;
}


template<class T_>
FVDenseM<T_> operator+(const FVDenseM<T_> &aa,const T_ &val)
{ 
   FVDenseM<T_> c(aa);
 // #pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<c.length; ++i)
      c.a[i] =aa.a[i]*val;
   return c;
}

template<class T_>
FVDenseM<T_> operator-(const FVDenseM<T_> &aa,const T_ &val)
{ 
   FVDenseM<T_> c(aa);
 // #pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<c.length; ++i)
      c.a[i] =aa.a[i]/val;
   return c;
}

template<class T_>
FVDenseM<T_> operator*(const FVDenseM<T_> &aa,const T_ &val)
{ 
   FVDenseM<T_> c(aa);
  //#pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<c.length; ++i)
      c.a[i] =aa.a[i]*val;
   return c;
}

template<class T_>
FVDenseM<T_> operator/(const FVDenseM<T_> &aa,const T_ &val)
{ 
   FVDenseM<T_> c(aa);
  //#pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<c.length; ++i)
      c.a[i] =aa.a[i]/val;
   return c;
}
/*--------- Matrix Vector operations ----------*/

 
// compute y=A*x
template<class T_>
void FVDenseM<T_>::Mult(const FVVect<T_> &x, FVVect<T_> &y) 
    {
register size_t pos_i; 
register size_t i,j; 


#ifdef _DEBUGS
    size_t err=(x.size()!=nb_cols)+2*(y.size()!=nb_rows);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif        
    y = static_cast<T_>(0);
    pos_i=0;
    for (i=0; i<nb_rows; i++)
        {
        pos_i=nb_cols*i;
        for (j=0; j<nb_cols; j++) 
            y[i] += a[pos_i+j] * x[j];
        }  
 
    }
// compute y=A^T*x
template<class T_>
void FVDenseM<T_>::TransMult(const FVVect<T_> &x, FVVect<T_> &y) 
    {
register size_t pos_j;        
#ifdef _DEBUGS
    size_t err=(x.size()!=nb_rows)+2*(y.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif         
    y = static_cast<T_>(0);
    //#pragma omp parallel for num_threads(nb_thread)   
    for (size_t i=0; i<nb_cols; i++)
        {
        pos_j=0;    
        for (size_t j=0; j<nb_rows; j++)
            y[i] += a[pos_j+i] * x[j];pos_j+=nb_cols;
        }
    } 
 /*--------- Matrix Matrix operations ----------*/
  template<class T_>    
void FVDenseM<T_>::show()
{
    
cout<<"==== MATRIX ===="<<nb_rows<<"X"<<nb_cols<<"===LENGTH==="<<length<<endl;    
for(size_t i=0;i<nb_rows;i++)
    {

    for(size_t j=0;j<nb_cols;j++)
        cout<<FVDenseM<T_>::getValue(i,j)<<" ";
    cout<<endl;
    }   
cout<<"==========================="<<endl;
}

/*--------- Matrix resolutions  Gauss ----------*/

template<class T_>
void  FVDenseM<T_>::Gauss(FVVect<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(u.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif   
size_t pos;
fv_float aux,pivot;
for(size_t k=0;k<nb_cols;k++)
    {
    // find the pivot 
    pos=k;aux=a[k*nb_cols+k];
    for(size_t i=k+1;i<nb_cols;i++)
        {
        if(a[i*nb_cols+k]*a[i*nb_cols+k]>= aux*aux)  {aux=a[i*nb_cols+k];pos=i;} 
        }
    // swap the two lines    
    for(size_t j=k;j<nb_cols;j++) 
        {
        aux=a[pos*nb_cols+j];  a[pos*nb_cols+j]=a[k*nb_cols+j]; a[k*nb_cols+j]=aux;
        }
    aux=u[pos];u[pos]=u[k];u[k]=aux; 
    // eliminate
    pivot=1/a[k*nb_cols+k];
    for(size_t i=k+1;i<nb_cols;i++)
        {    
        aux=a[i*nb_cols+k]*pivot;    
        for(size_t j=k;j<nb_cols;j++)    
            {
            a[i*nb_cols+j]-=aux*a[k*nb_cols+j];
            } 
        u[i]-=aux*u[k];    
        }
    }
  // do the backward substitution   
pos=length-nb_rows;
for(size_t i=0;i<nb_rows;i++)
    {
    aux=0.;
    size_t ii=nb_rows-i-1;
 //   #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)
    for(size_t j=ii+1;j<nb_cols;j++)
        aux+=a[pos+j]*u[j]; 
    u[ii]-=aux;u[ii]/=a[pos+ii];
    pos-=nb_rows;
    } 
}
template<class T_>
void  FVDenseM<T_>::Gauss(FVPoint2D<fv_float> &u)
{
fv_float det=a[0]*a[3]-a[1]*a[2];
#ifdef _DEBUGS
  if (det==0.0) cout<<" Warning, singular matrix"<<endl; 
#endif   
FVPoint2D<fv_float> f(u);
u.x=(f.x*a[3]-f.y*a[1])/det;
u.y=(f.y*a[0]-f.x*a[2])/det;
}
template<class T_>
void  FVDenseM<T_>::Gauss(FVPoint3D<fv_float> &u)
{
fv_float det=a[0]*(a[4]*a[8]-a[7]*a[5])-
           a[3]*(a[1]*a[8]-a[7]*a[2])+
           a[6]*(a[1]*a[5]-a[4]*a[2]);
#ifdef _DEBUGS
  if (det==0.0) cout<<" Warning, singular matrix"<<endl; 
#endif 
FVPoint3D<fv_float> f(u);
u.x=f.x*(a[4]*a[8]-a[7]*a[5])-
    f.y*(a[1]*a[8]-a[7]*a[2])+
    f.z*(a[1]*a[5]-a[4]*a[2]);u.x/=det;
u.y=f.x*(a[3]*a[8]-a[6]*a[5])-
    f.y*(a[0]*a[8]-a[6]*a[2])+
    f.z*(a[0]*a[5]-a[3]*a[2]);u.y/=-det;
u.z=f.x*(a[3]*a[7]-a[6]*a[4])-
    f.y*(a[0]*a[7]-a[6]*a[1])+
    f.z*(a[0]*a[4]-a[3]*a[1]);u.z/=det;
}
template<class T_>
void  FVDenseM<T_>::Gauss(FVPoint4D<fv_float> &u)
{
FVVect<fv_float> f(4);
f[0]=u.x;f[1]=u.y;f[2]=u.z;f[3]=u.t;
FVDenseM<T_>::Gauss(f);
u.x=f[0];u.y=f[1];u.z=f[2];u.t=f[3];
}
    
    
    
    
/*--------- Matrix resolutions  LU ----------*/
/* 
  //--------- BASIC VERSION  -----------
void FVDenseM<T_>::LUFactorize() const 
{
register fv_float sum; 
for(size_t k=0;k<nb_rows;k++)
    {
    for(size_t j=0;j<k;j++)
        {
         sum=0;
         for(size_t i=0;i<j;i++) sum+=a[nb_cols*k+i]*a[nb_cols*i+j];
         a[nb_cols*k+j]=(a[nb_cols*k+j]-sum)/a[nb_cols*j+j];
        }
    for(size_t j=k;j<nb_cols;j++)
        {
         sum=0;
         for(size_t i=0;i<k;i++) sum+=a[nb_cols*k+i]*a[nb_cols*i+j];
         a[nb_cols*k+j]-=sum;
        }   
    }    
}   */

// first optimized version
template<class T_>
void FVDenseM<T_>::LUFactorize() 
{
register fv_float sum; 
register size_t pos_k;
for(size_t k=0;k<nb_rows;k++)
    {
    pos_k= nb_cols*k;   
    for(size_t j=0;j<k;j++)
        {
         sum=0.;
      //   #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)  
         for(size_t i=0;i<j;i++) sum+=a[pos_k+i]*a[nb_cols*i+j];
         a[pos_k+j]=(a[pos_k+j]-sum)/a[nb_cols*j+j];
        }
    for(size_t j=k;j<nb_cols;j++)
        {
         sum=0.;
      //   #pragma omp parallel for reduction(+:sum) num_threads(nb_thread) 
         for(size_t i=0;i<k;i++) {sum+=a[pos_k+i]*a[nb_cols*i+j];}
         a[pos_k+j]-=sum;
        }   
    }    
}
template<class T_>
void FVDenseM<T_>::ForwardSubstitution(FVVect<fv_float> &u) 
{
register fv_float sum;    
register size_t pos_i;
#ifdef _DEBUGS
    size_t err=(u.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
pos_i=0;
for(size_t i=0;i<nb_rows;i++)
    {
    sum=0.;
  //  #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)
    for(size_t j=1;j<=i;j++) // WARNING we need to allow "j=-1"    
                             // because we use unsigned int, we shift the indexes of 1 
                             // So we start at 1 til j ON PURPOSE
        sum+=a[pos_i+j-1]*u[j-1];
    u[i]-=sum;
    pos_i+=nb_cols;
    }
}
// forward substitution 
template<class T_>
void FVDenseM<T_>::ForwardSubstitution(FVPoint2D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=2);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y-=u.x*a[2];
}
template<class T_>
void FVDenseM<T_>::ForwardSubstitution(FVPoint3D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=3);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y-=u.x*a[3];u.z-=a[6]*u.x+a[7]*u.y;
}
template<class T_>
void FVDenseM<T_>::ForwardSubstitution(FVPoint4D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=4);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y-=u.x*a[4];u.z-=a[8]*u.x+a[9]*u.y;u.t-=a[12]*u.x+a[13]*u.y+a[14]*u.z;
}
// forward substitution 
template<class T_>
void FVDenseM<T_>::BackwardSubstitution(FVVect<fv_float> &u) 
{
register fv_float sum;
register size_t pos_ii;
#ifdef _DEBUGS
    size_t err=(u.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
pos_ii=length-nb_rows;
for(size_t i=0;i<nb_rows;i++)
    {
    sum=0.;
    size_t ii=nb_rows-i-1;
 //   #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)
    for(size_t j=ii+1;j<nb_cols;j++)
        sum+=a[pos_ii+j]*u[j]; 
    u[ii]-=sum;u[ii]/=a[pos_ii+ii];
    pos_ii-=nb_rows;
    }
}
template<class T_>
void FVDenseM<T_>::BackwardSubstitution(FVPoint2D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=2);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y/=a[3]; u.x=(u.x-a[1]*u.y)/a[0];
}
template<class T_>
void FVDenseM<T_>::BackwardSubstitution(FVPoint3D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=3);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.z/=a[8];u.y=(u.y-a[5]*u.z)/a[4];u.x=(u.x-a[1]*u.y-a[2]*u.z)/a[0];
}
template<class T_>
void FVDenseM<T_>::BackwardSubstitution(FVPoint4D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=4);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.t/=a[15];u.z=(u.z-a[11]*u.t)/a[10];u.y=(u.y-a[6]*u.z-a[7]*u.t)/a[5];u.x=(u.x-a[1]*u.y-a[2]*u.z-a[3]*u.t)/a[0];
}
//
  //--------- Pivoting  VERSION  -----------
  //
template<class T_>  
void FVDenseM<T_>::LUFactorizePivoting() 
{
register fv_float sum,maxval;
size_t row,permut;
for(size_t i=0;i<nb_rows;i++) row_perm[i]=i;
for(size_t k=0;k<nb_rows;k++)
    {
    row=k; maxval=0;
    for(size_t i=k;i<nb_rows;i++)
        if(maxval< abs(a[nb_cols*row_perm[i]+k])) {maxval=abs(a[nb_cols*row_perm[i]+k]);row=i;}
    permut=row_perm[k]; row_perm[k]=row_perm[row];row_perm[row]=permut;    
    for(size_t j=0;j<k;j++)
        {
         sum=0;
         for(size_t i=0;i<j;i++) sum+=a[nb_cols*row_perm[k]+i]*a[nb_cols*row_perm[i]+j];
         a[nb_cols*row_perm[k]+j]=(a[nb_cols*row_perm[k]+j]-sum)/a[nb_cols*row_perm[j]+j];
        }
    for(size_t j=k;j<nb_cols;j++)
        {
         sum=0;
         for(size_t i=0;i<k;i++) sum+=a[nb_cols*row_perm[k]+i]*a[nb_cols*row_perm[i]+j];
         a[nb_cols*row_perm[k]+j]-=sum;
        }   
    }    
}   

template<class T_>
void FVDenseM<T_>::ForwardSubstitutionPivoting(FVVect<fv_float> &u) 
{
register fv_float sum;    
#ifdef _DEBUGS
    size_t err=(u.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
FVVect<fv_float> f(u);    
for(size_t i=0;i<nb_rows;i++)
    {
    sum=0.;
  //  #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)
    for(size_t j=1;j<=i;j++) // WARNING we need to allow "j=-1"    
                             // because we use unsigned int, we shift the indexes of 1 
                             // So we start at 1 til j ON PURPOSE
        sum+=a[nb_cols*row_perm[i]+j-1]*u[j-1];
    u[i]=f[row_perm[i]]-sum;
    }
}
// forward substitution 
template<class T_>
void FVDenseM<T_>::ForwardSubstitutionPivoting(FVPoint2D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=2);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y-=u.x*a[2];
}
template<class T_>
void FVDenseM<T_>::ForwardSubstitutionPivoting(FVPoint3D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=3);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y-=u.x*a[3];u.z-=a[6]*u.x+a[7]*u.y;
}
template<class T_>
void FVDenseM<T_>::ForwardSubstitutionPivoting(FVPoint4D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=4);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y-=u.x*a[4];u.z-=a[8]*u.x+a[9]*u.y;u.t-=a[12]*u.x+a[13]*u.y+a[14]*u.z;
}
// forward substitution 
template<class T_>
void FVDenseM<T_>::BackwardSubstitutionPivoting(FVVect<fv_float> &u) 
{
register fv_float sum;
#ifdef _DEBUGS
    size_t err=(u.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
FVVect<fv_float> f(u);   
size_t ii;
for(size_t i=0;i<nb_rows;i++)
    {
    sum=0.;
    ii=nb_rows-i-1;
 //   #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)
    for(size_t j=ii+1;j<nb_cols;j++)
        sum+=a[nb_cols*row_perm[ii]+j]*u[j]; 
    u[ii]=f[row_perm[ii]]-sum;
    u[ii]/=a[nb_cols*row_perm[ii]+ii];
    }
}
template<class T_>
void FVDenseM<T_>::BackwardSubstitutionPivoting(FVPoint2D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=2);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.y/=a[3]; u.x=(u.x-a[1]*u.y)/a[0];
}
template<class T_>
void FVDenseM<T_>::BackwardSubstitutionPivoting(FVPoint3D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=3);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.z/=a[8];u.y=(u.y-a[5]*u.z)/a[4];u.x=(u.x-a[1]*u.y-a[2]*u.z)/a[0];
}
template<class T_>
void FVDenseM<T_>::BackwardSubstitutionPivoting(FVPoint4D<fv_float> &u) 
{
#ifdef _DEBUGS
    size_t err=(nb_cols!=4);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif 
u.t/=a[15];u.z=(u.z-a[11]*u.t)/a[10];u.y=(u.y-a[6]*u.z-a[7]*u.t)/a[5];u.x=(u.x-a[1]*u.y-a[2]*u.z-a[3]*u.t)/a[0];
}



//
//--------- Matrix resolutions  QR ----------
//
template<class T_>
void FVDenseM<T_>::QRFactorize(FVDenseM<fv_float> &q)   
{
    // Caution: q is the Transpose Matrix of the decomposition
 #ifdef _DEBUGS
    size_t err=(q.nb_rows!=nb_rows)+2*(q.nb_cols!=nb_rows);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif  
FVVect<fv_float> v(nb_rows);
register fv_float norm,aux,ps;
for(size_t k=0;k<q.length;k++) q.a[k]=0;
for(size_t j=0;j<nb_rows;j++) q.a[j*(nb_rows+1)]=1;
for(size_t k=0;k<nb_cols;k++)
    {
    norm=0;    
    for(size_t i=k;i<nb_rows;i++)
        {v[i]=a[nb_cols*i+k];norm+=v[i]*v[i];}  
    aux=v[k]*v[k];    
    if(v[k]<0) v[k]-=sqrt(norm); else v[k]+=sqrt(norm);
    norm=sqrt(norm+v[k]*v[k]-aux); // the new norm
    for(size_t i=k;i<nb_rows;i++) v[i]/=norm; // the normalize householder vector
    for(size_t j=k;j<nb_cols;j++)
        {
        ps=0;    
        for(size_t i=k;i<nb_rows;i++) ps+=v[i]*a[nb_cols*i+j];
        for(size_t i=k;i<nb_rows;i++) a[nb_cols*i+j]-=2*ps*v[i];
        }
    for(size_t j=0;j<nb_rows;j++)        
        {
        ps=0;    
        for(size_t i=k;i<nb_rows;i++) ps+=v[i]*q.a[nb_rows*i+j];
        for(size_t i=k;i<nb_rows;i++) q.a[nb_rows*i+j]-=2*ps*v[i];        
        }
    }
}

template<class T_>
void FVDenseM<T_>::PartialBackwardSubstitution( FVVect<fv_float> & u)
{
register fv_float sum;
register size_t pos_ii;
#ifdef _DEBUGS
    size_t err=(u.size()!=nb_rows); 
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<" index out of range, error code="<<err<<endl;
#endif 
pos_ii=nb_cols*(nb_cols-1);
// i can remove this line since i do no use the element >=nb_cols
for(size_t i=nb_cols;i<nb_rows;i++) u[i]=0;//only the first nb_cols elements are correct
for(size_t i=0;i<nb_cols;i++)
    {
    sum=0.;
    size_t ii=nb_cols-i-1;
 //   #pragma omp parallel for reduction(+:sum) num_threads(nb_thread)
    for(size_t j=ii+1;j<nb_cols;j++)
        sum+=a[pos_ii+j]*u[j]; 
    u[ii]-=sum;u[ii]/=a[pos_ii+ii];
    pos_ii-=nb_cols;
    }  
}
#endif // define _FVDenseM

 
 
