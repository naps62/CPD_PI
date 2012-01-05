#ifndef __FVIO_H
#define __FVIO_H


#include <string>
#include <iostream>
#include <iomanip>
#include<XML.h>
#include<FVVect.h>
#include<FVPoint2D.h>
#include<FVPoint3D.h>
#include<FVLib_config.h>


  

class FVio
{
public:

    FVio();
    FVio(const char *, int );
    ~FVio();
    void setTime(double &time){_time=time;}
    void setName(string &name){_name=name;} 
    void open(const char *, int );
    void close();
    void showXML(){cout << _xml<<endl;}  // for internal purpose to check the string
    void put(FVVect <double> &, const double time=0., const string &name="noname");      
    void put(FVVect <double> &, FVVect <double> &, const double time=0., const string &name="noname");  
    void put(FVVect <FVPoint2D<double> >&, const double time=0., const string &name="noname");      
    void put(FVVect <double> &, FVVect <double> &,FVVect <double> &,  const double time=0., const string &name="noname");
    void put(FVVect <FVPoint3D<double> >&, const double time=0., const string &name="noname");     
    // one vector
    size_t get(FVVect <double> &u)
         {return(FVio::get(u,_time,_name));}
    size_t get(FVVect <double> &u,  double &time )
         {return(FVio::get(u,time,_name));}    
    size_t get(FVVect <double> &,  double &, string &);
    // two vectors
    size_t get(FVVect <double> &u, FVVect <double> &v)
         {return(FVio::get(u,v,_time,_name));}
    size_t get(FVVect <double> &u, FVVect <double> &v, double &time)
         {return(FVio::get(u,v,time,_name));}  
    size_t get(FVVect <double> &, FVVect <double> &, double &, string &);
          // with FVPOINT2D
    size_t get(FVVect <FVPoint2D<double> > &u)
         {return(FVio::get(u,_time,_name));}
    size_t get(FVVect <FVPoint2D<double> >&u, double &time)
         {return(FVio::get(u,time,_name));}  
    size_t get(FVVect <FVPoint2D<double> >&, double &, string &);    
    // three vectors
    size_t get(FVVect <double> &u, FVVect <double> &v, FVVect <double> &w)
         {return(FVio::get(u,v,w,_time,_name));} 
    size_t get(FVVect <double> &u, FVVect <double> &v, FVVect <double> &w, double &time)   
         {return(FVio::get(u,v,w,time,_name));}     
    size_t get(FVVect <double> &, FVVect <double> &, FVVect <double> &, double &, string &);    
           // with FVPOINT3D
    size_t get(FVVect <FVPoint3D<double> > &u)
         {return(FVio::get(u,_time,_name));}
    size_t get(FVVect <FVPoint3D<double> >&u, double &time)
         {return(FVio::get(u,time,_name));}  
    size_t get(FVVect <FVPoint3D<double> >&, double &, string &);  
    size_t getNbVect(){return(_nbvec);}
private:
    string _xml,_name;
    SparseXML _spxml;
    double _time;
    size_t _nbvec,_sizevec;
    bool _is_open; 
    ofstream        _of;
    string          _field_name;
    size_t          _access;

};



#endif     // end of ifndef __FVIO_H
