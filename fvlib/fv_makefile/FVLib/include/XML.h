// ------ XML.h ------
// S. CLAIN 2010/12
#ifndef _XML 
#define _XML

#include <string>
#include<cstdlib>
#include <map>
#include <list>
#include <stack>
#include <fstream>
#include <iostream>
#include <FVLib_config.h>
using namespace std;




class SparseXML 
{
public:

          //constructor
  SparseXML();
  SparseXML(const char *,string &); // of with a file
  SparseXML(string &); // I have a string
        //destructor
  ~SparseXML(){;}
  // member
  void readXML(const char *,string &);
  int openBalise(); // open the next Balise
  int openBalise(const string &, size_t level=0);// find  the  open balise element from the current position  
  int openBalise(const char *, size_t level=0);// find  the  open balise element from the current position  
  int closeBalise();  //close the next Balise
  int closeBalise(const string &, size_t level=0);// find  the close balise element from the current position  
  int closeBalise(const char *, size_t level=0);// find  the close balise element from the current position  
  void data();
  size_t getPosition(){return _data_position ;}
  size_t getLength(){return _data_length;}
  size_t getLevel(){return _level;}
  string getElement(){return _element;}
  StringMap getAttribute(){ return _attribute;}
  
private:
void makeMap(string &);  
void removeComment();
// variable list
bool _empty_element;
size_t _level;
size_t  _current_position, _data_position,_data_length;
string * _xml_string;  
string _key;
string _value;
string _element;
StringMap _attribute;
};

#endif // define _XML