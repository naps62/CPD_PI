// ------ FVSparseMatrice.h ------
// S. CLAIN 2011/07
#ifndef _FVSparseM
#define _FVSparseM

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <FVVect.h>
#define NO_INDEX ((size_t) -1)
class SparseNode
{
public:
    size_t index;
    size_t pos;
    SparseNode(){index=0;pos=0;}
};

typedef std::vector<SparseNode> Tab_index;
template<class T_> class FVSparseM 
{
private:
    size_t  exist( Tab_index *tab,size_t loc);  
public: 
size_t nb_cols,nb_rows,length; // number of rows and columns  , length of the matrix 
std::vector<Tab_index *> row,col; // index table
std::vector<T_>  a;  // the place for the matrice

    FVSparseM();
    ~FVSparseM();
    FVSparseM(size_t );
    FVSparseM(size_t , size_t );
    FVSparseM(const FVSparseM<T_> &m);
    size_t getNbColumns() {return nb_cols;}
    size_t getNbRows(){ return nb_rows;}
    size_t getLength(){ return length;} 
    void resize(size_t );
    void resize(size_t , size_t );
    void setValue(size_t i, size_t j, const T_ &val);
    void addValue(size_t i, size_t j, const T_ &val);    
    void resizeAndsetValue(size_t i, size_t j, const T_ &val);    
    T_ getValue(size_t i, size_t j)  ;
    void show();
    //FVSparseM<T_> & operator=(const T_ &x);
    //FVSparseM<T_> & operator+=(const FVSparseM<T_> &m);
    //FVSparseM<T_> & operator-=(const FVSparseM<T_> &m);
    FVSparseM<T_> & operator/=(const T_ &val);
    FVSparseM<T_> & operator*=(const T_ &val);
    //FVSparseM<T_> & operator+=(const T_ &val);
    //FVSparseM<T_> & operator-=(const T_ &val);
    //void horizontalMerge(FVSparseM<T_> &m, const FVSparseM<T_> &m,);
    //void verticalMerge(FVSparseM<T_> &m, const FVSparseM<T_> &m,);    
    //FVVect<T_> getColumn(size_t j) const;
    //FVVect<T_> getRow(size_t i) const
    void Mult(const FVVect<T_> &, FVVect<T_> &) const;
    void TransMult(const FVVect<T_> &, FVVect<T_> &) const  ; 
    
 
};





//construct an empty matrix
template<class T_>
FVSparseM<T_>::FVSparseM()
    {
    nb_rows = nb_cols = 0,length=0;
    }
    // destructor: free the memory
template<class T_>
FVSparseM<T_>::~FVSparseM()
    {
    nb_rows = nb_cols = 0,length=0;
    for(size_t i=0;i<row.size();i++)
        if((row[i])) delete row[i];
    for(size_t j=0;j<col.size();j++)
        if((col[j])) delete col[j];
    }    
template<class T_>    
void FVSparseM<T_>::resize(size_t size)
    {
    nb_rows = nb_cols = size;
    length=0;
    a.resize(length);
    row.resize(nb_rows,NULL);
    col.resize(nb_rows,NULL);      
    }
    
template<class T_>   
void FVSparseM<T_>::resize(size_t nr, size_t nc)
    {
    nb_rows = nr;nb_cols = nc;
    length=0;
    a.resize(length);
    row.resize(nb_rows,NULL);
    col.resize(nb_rows,NULL);     
    }
    
// construct a square matrix
template<class T_>
FVSparseM<T_>::FVSparseM(size_t size)
    {
     FVSparseM<T_>::resize(size);  
    }
// construct a rectangular matrix
template<class T_>
FVSparseM<T_>::FVSparseM(size_t nr, size_t nc)
    {
    FVSparseM<T_>::resize(nr,nc);
    }
// copy dense matrix    
template<class T_>
FVSparseM<T_>::FVSparseM(const FVSparseM<T_> &m)
{
   FVSparseM<T_>::resize(m.nb_rows,m.nb_cols);
   row=m.row;
   col=m.col;
   a = m.a;
}
// find a node in a line or a column
template<class T_>
size_t FVSparseM<T_>::exist( Tab_index *tab,size_t loc)
{
for(size_t i=0;i<tab->size();i++)
    {
     if ((*tab)[i].index==loc) return(i);   
    }
return NO_INDEX;    
}
//   set value
template<class T_>
void FVSparseM<T_>::setValue(size_t i, size_t j, const T_ &val)
    {
    size_t jj;    
    if ((i> nb_rows) || (j> nb_cols)) 
       {cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range,"<<endl;}
    //cout<<"i="<<i<<" j="<<j<<endl; fflush(stdout);   
    if (row[i]==NULL) row[i]=new Tab_index;     
    if (col[j]==NULL) col[j]=new Tab_index;    
    jj=exist(row[i],j);
    if(jj!=NO_INDEX) a[(*(row[i]))[jj].pos]=val; // the node still exist
    else  // we have to create the node for the two index set
        {
        SparseNode no;
        a.push_back(val);
        length=a.size();
        no.index=j;no.pos=length-1;
        row[i]->push_back(no);
        no.index=i;
        col[j]->push_back(no);
        } 
    }
 //   add value
template<class T_>
void FVSparseM<T_>::addValue(size_t i, size_t j, const T_ &val)
    {
    size_t jj;    
    if ((i> nb_rows) || (j> nb_cols)) 
       {cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range,"<<endl;}
    //cout<<"i="<<i<<" j="<<j<<endl; fflush(stdout);   
    if (row[i]==NULL) row[i]=new Tab_index;     
    if (col[j]==NULL) col[j]=new Tab_index;    
    jj=exist(row[i],j);
    if(jj!=NO_INDEX) a[(*(row[i]))[jj].pos]+=val; // the node still exist
    else  // we have to create the node for the two index set
        {
        SparseNode no;
        a.push_back(val);
        length=a.size();
        no.index=j;no.pos=length-1;
        row[i]->push_back(no);
        no.index=i;
        col[j]->push_back(no);
        } 
    }
 //  resize  and set value    
template<class T_>    
void FVSparseM<T_>::resizeAndsetValue(size_t i, size_t j, const T_ &val)    
    {
    size_t nc,nr;    
    if(i> nb_rows) nr=i; else nr=nb_rows;
    if(j> nb_cols) nc=j; else nc=nb_cols;
    FVSparseM<T_>::resize(nr,nc);
     setValue(i,j,val);
    }
 //   get value    
template<class T_>    
T_ FVSparseM<T_>::getValue(size_t i, size_t j)  
    {
    size_t jj;    
    //cout<<"i="<<i<<" j="<<j<<endl; fflush(stdout);
    if ((i> nb_rows) || (j> nb_cols)) return static_cast<T_>(0);  //outside of the matrix
    if (row[i]==NULL) return static_cast<T_>(0);  
    jj=exist(row[i],j);
    if(jj==NO_INDEX) return static_cast<T_>(0);
    else
        return(a[(*(row[i]))[jj].pos]);
     
    } 
 template<class T_>    
void FVSparseM<T_>::show()
{
    
cout<<"==== MATRIX ===="<<nb_rows<<"X"<<nb_cols<<"===LENGTH==="<<length<<endl;    
for(size_t i=0;i<nb_rows;i++)
    {

    for(size_t j=0;j<nb_cols;j++)
        cout<<FVSparseM<T_>::getValue(i,j)<<" ";
    cout<<endl;
    }   
cout<<"==========================="<<endl;
cout<<"TAB_VALUE"<<endl;
for(size_t i=0;i<a.size();i++) cout<<a[i]<<" ";
cout<<endl;
}

/*--------- Matrix Vector operations ----------*/

template<class T_>
FVSparseM<T_> & FVSparseM<T_>::operator/=(const T_ &val)
{
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] /= val;
   return *this;
}

template<class T_>
FVSparseM<T_> & FVSparseM<T_>::operator*=(const T_ &val)
{
 // #pragma omp parallel for num_threads(nb_thread)
   for (size_t i=0; i<length; i++)
      a[i] *= val;
   return *this;
}
 
// compute y=A*x
template<class T_>
void FVSparseM<T_>::Mult(const FVVect<T_> &x, FVVect<T_> &y) const
    {
#ifdef _DEBUGS
    size_t err=(x.size()!=nb_cols)+2*(y.size()!=nb_rows);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif    
    y = static_cast<T_>(0);
//#pragma omp parallel for num_threads(nb_thread)
    for (size_t i=0; i<nb_rows; i++)
        {
        if(!row[i]) continue; // no line i so y[i)=0     
        for (size_t jj=0; jj<(row[i])->size(); jj++)
            {
             //cout<<"i="<<i<<" j="<<  (*(row[i]))[jj].index<<" val="<< a[(*(row[i]))[jj].pos]<<endl;
             y[i] += a[(*(row[i]))[jj].pos] * x[(*(row[i]))[jj].index];
            }
        }
    }
// compute y=A^T*x
template<class T_>
void FVSparseM<T_>::TransMult(const FVVect<T_> &x, FVVect<T_> &y) const
    {
#ifdef _DEBUGS
    size_t err=(x.size()!=nb_rows)+2*(y.size()!=nb_cols);
    if(err) cout<<" in file "<<__FILE__<<", line "<<__LINE__<<"index out of range, error code="<<err<<endl;
#endif         
    y = static_cast<T_>(0);
//#pragma omp parallel for num_threads(nb_thread)    
    for (size_t j=0; j<nb_cols; j++)
        {
        if(!col[j]) continue; // no column j, y[i]=0     
        for (size_t ii=0; ii<col[j]->size(); ii++)
            y[j] += a[(*(col[j]))[ii].pos] * x[(*(col[j]))[ii].index];
        }
    } 
 /*--------- Matrix Matrix operations ----------*/
#endif // define _FVSparseM

 
 
