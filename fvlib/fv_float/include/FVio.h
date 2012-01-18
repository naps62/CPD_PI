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
#include "MFVLib_config.h"

  

class FVio
{
public:

    FVio();
    FVio(const char *, int );
    ~FVio();
    void setTime(fv_float &time){_time=time;}
    void setName(string &name){_name=name;} 
    void open(const char *, int );
    void close();
    void showXML(){cout << _xml<<endl;}  // for internal purpose to check the string
    void put(FVVect <fv_float> &, const fv_float time=0., const string &name="noname");      
    void put(FVVect <fv_float> &, FVVect <fv_float> &, const fv_float time=0., const string &name="noname");  
    void put(FVVect <FVPoint2D<fv_float> >&, const fv_float time=0., const string &name="noname");      
    void put(FVVect <fv_float> &, FVVect <fv_float> &,FVVect <fv_float> &,  const fv_float time=0., const string &name="noname");
    void put(FVVect <FVPoint3D<fv_float> >&, const fv_float time=0., const string &name="noname");     
    // one vector
    size_t get(FVVect <fv_float> &u)
         {return(FVio::get(u,_time,_name));}
    size_t get(FVVect <fv_float> &u,  fv_float &time )
         {return(FVio::get(u,time,_name));}    
    size_t get(FVVect <fv_float> &,  fv_float &, string &);
    // two vectors
    size_t get(FVVect <fv_float> &u, FVVect <fv_float> &v)
         {return(FVio::get(u,v,_time,_name));}
    size_t get(FVVect <fv_float> &u, FVVect <fv_float> &v, fv_float &time)
         {return(FVio::get(u,v,time,_name));}  
    size_t get(FVVect <fv_float> &, FVVect <fv_float> &, fv_float &, string &);
          // with FVPOINT2D
    size_t get(FVVect <FVPoint2D<fv_float> > &u)
         {return(FVio::get(u,_time,_name));}
    size_t get(FVVect <FVPoint2D<fv_float> >&u, fv_float &time)
         {return(FVio::get(u,time,_name));}  
    size_t get(FVVect <FVPoint2D<fv_float> >&, fv_float &, string &);    
    // three vectors
    size_t get(FVVect <fv_float> &u, FVVect <fv_float> &v, FVVect <fv_float> &w)
         {return(FVio::get(u,v,w,_time,_name));} 
    size_t get(FVVect <fv_float> &u, FVVect <fv_float> &v, FVVect <fv_float> &w, fv_float &time)   
         {return(FVio::get(u,v,w,time,_name));}     
    size_t get(FVVect <fv_float> &, FVVect <fv_float> &, FVVect <fv_float> &, fv_float &, string &);    
           // with FVPOINT3D
    size_t get(FVVect <FVPoint3D<fv_float> > &u)
         {return(FVio::get(u,_time,_name));}
    size_t get(FVVect <FVPoint3D<fv_float> >&u, fv_float &time)
         {return(FVio::get(u,time,_name));}  
    size_t get(FVVect <FVPoint3D<fv_float> >&, fv_float &, string &);  
    size_t getNbVect(){return(_nbvec);}
private:
    string _xml,_name;
    SparseXML _spxml;
    fv_float _time;
    size_t _nbvec,_sizevec;
    bool _is_open; 
    ofstream        _of;
    string          _field_name;
    size_t          _access;

};



#endif     // end of ifndef __FVIO_H

