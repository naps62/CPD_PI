// ------ FVVect.h ------
// S. CLAIN 2011/07
#ifndef _FVVect 
#define _FVVect
#include "FVLib_config.h"
#include "FVGlobal.h"
#include <valarray>
#include <iostream>
using std::cout;
using std::endl;
using std::valarray;
template<class T_>  class FVVect : public valarray<T_>
{
    public:
    using valarray<T_>::size;
    using valarray<T_>::resize;   
    
    FVVect();
    FVVect(const  size_t );
    FVVect(const FVVect<T_>& v);
    FVVect<T_> & operator=(const T_ &a);
    FVVect<T_> & operator+=(const FVVect<T_> &x);
    FVVect<T_> & operator+=(const T_ &a);
    FVVect<T_> & operator-=(const FVVect<T_> &x);
    FVVect<T_> & operator-=(const T_ &a);
    FVVect<T_> &operator*=(const T_ &a);
    FVVect<T_> &operator/=(const T_ &a);
    T_ beginElement(){pos=0;if(pos< this->size()) return((*this)[0]);else return(NULL);}
    T_ nextElement(){if(pos<this->size()) return((*this)[pos++]);else return(NULL);}  
    void show();
private:
size_t  pos;

};

///////////////////////////////////////////////////////////////////////////////
//                         I M P L E M E N T A T I O N                       //
///////////////////////////////////////////////////////////////////////////////
template<class T_>
FVVect<T_>::FVVect() : valarray<T_>(0){;}
template<class T_>
FVVect<T_>::FVVect(const  size_t n) : valarray<T_>(n){;}
template<class T_>
FVVect<T_>::FVVect(const FVVect<T_>& v) : valarray<T_>(v){;}

/*-------- operators overload -------*/ 
template<class T_>
FVVect<T_> &FVVect<T_>::operator=(const T_ &a)
{
//#pragma omp parallel for num_threads(nb_thread)  
   for (size_t i=0; i<size(); ++i)
      (*this)[i] = a;
   return *this;
}

template<class T_>
FVVect<T_> &FVVect<T_>::operator+=(const FVVect<T_> &x)
{
//#pragma omp parallel for num_threads(nb_thread) 
   for (size_t i=0; i<size(); ++i)
      (*this)[i] += x[i];
   return *this;
}


template<class T_>
FVVect<T_> &FVVect<T_>::operator+=(const T_ &a)
{
//#pragma omp parallel for num_threads(nb_thread) 
   for (size_t i=0; i<size(); ++i)
      (*this)[i] += a;
   return *this;
}


template<class T_>
FVVect<T_> &FVVect<T_>::operator-=(const FVVect<T_> &x)
{
//#pragma omp parallel for num_threads(nb_thread) 
   for (size_t i=0; i<size(); ++i)
      (*this)[i] -= x[i];
   return *this;
}


template<class T_>
FVVect<T_> &FVVect<T_>::operator-=(const T_ &a)
{
//#pragma omp parallel for num_threads(nb_thread) 
   for (size_t i=0; i<size(); ++i)
      (*this)[i] -= a;
   return *this;
}


template<class T_>
FVVect<T_> &FVVect<T_>::operator*=(const T_ &a)
{
//#pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<size(); ++i)
      (*this)[i] *= a;
   return *this;
}


template<class T_>
FVVect<T_> &FVVect<T_>::operator/=(const T_ &a)
{
//#pragma omp parallel for num_threads(nb_thread)  
   for (size_t i=0; i<size(); ++i)
      (*this)[i] /= a;
   return *this;
}

template<class T_>
FVVect<T_> operator+(const FVVect<T_> &x, const FVVect<T_> &y)
{
   FVVect<T_> v(x);
  //#pragma omp parallel for num_threads(nb_thread)   
   for (size_t i=0; i<x.size(); ++i)
      v[i] += y[i];
   return v;
}

template<class T_>
FVVect<T_> operator-(const FVVect<T_> &x, const FVVect<T_> &y)
{
   FVVect<T_> v(x);
   //#pragma omp parallel for num_threads(nb_thread)  
   for (size_t i=0; i<x.size(); ++i)
      v[i] -= y[i];
   return v;
}
template<class T_>
FVVect<T_> operator*(const T_ &a, const FVVect<T_> &x)
{
   FVVect<T_> v(x);
   //#pragma omp parallel for num_threads(nb_thread)  
   for (size_t i=0; i<x.size(); ++i)
      v[i] *= a;
   return v;
}

template<class T_>
void FVVect<T_>::show()
{
    cout<<"VECT====size"<<this->size()<<endl;
    for(size_t i=0;i<this->size();i++)
        cout<<(*this)[i]<<"  ";
    cout<<endl;
}
#endif // define _FVVect
 
