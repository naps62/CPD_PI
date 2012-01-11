// ------ Parameter.h ------
// S. CLAIN 2010/12
#ifndef _Parameter 
#define _Parameter

#include <cstdlib>
#include <XML.h>
using namespace std;

class Parameter
{
  
public:
          //constructor
Parameter();
Parameter(const char *); // take the first parameter section
Parameter(string &){cout<<"under construction"<<endl;} // take the first parameter section
        //destructor
~Parameter(){;}
  // member  
double getDouble(const string &key)
    { 
      string value;
      value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;}
      return(atof(value.c_str()));
    }  
int getInteger(const string &key)
    {
      string value;
      value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;}
      return(atoi(value.c_str()));
    }
size_t getUnsigned(const string &key)
    {
      string value;
     value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;} 
      return((unsigned) atoi( value.c_str()));     
    }
string getString(const string &key)
    {
      string value;
     value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;}       
      return( value);
    }
double getDouble(const char *keyname)
    {
      string key(keyname),value; 
      value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;}
      return(atof(value.c_str()));      
    }  
int getInteger(const char *keyname)
    {
      string key(keyname),value; 
      value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;}
      return(atoi(value.c_str()));      
    }
size_t getUnsigned(const char *keyname)
    {
      string key(keyname),value; 
     value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;} 
      return((unsigned) atoi( value.c_str()));      
    }
string getString(const char *keyname)
    {
      string key(keyname),value; 
      value=_param[key];
      if(value.empty())
           {cout<<key<<" is not a valid key"<<endl;}       
      return( value);     
    } 
private:
StringMap _param;
}; 

#endif // define _Parameter