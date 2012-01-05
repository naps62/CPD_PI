 // ------ Table.h ------
// S. CLAIN 2010/12
#ifndef _Table 
#define _Table

#include<XML.h>
#include<valarray>
#include <cstdlib>
using namespace std;
class Table
{
public:
          //constructor
Table();
Table(const char *,const char *); //take the function with given name
Table(string &,const char *){cout<<"under construction"<<endl;}//take the parameter section with name
        //destructor
~Table(){;}
  // member     
size_t  getNbPoints1(){return _nb_pts1;}
size_t  getNbPoints2(){return _nb_pts2;}
size_t  getNbPoints3(){return _nb_pts3;}
double getMin1(){return _min1;}
double getMin2(){return _min2;}
double getMin3(){return _min3;}
double getMax1(){return _max1;}
double getMax2(){return _max2;}
double getMax3(){return _max3;}
string getVar1(){return _var1;}
string getVar2(){return _var2;}
string getVar3(){return _var3;}
void printArray()
       {
         if(!_table.size()) {cout<<"empty array"<<endl;return;}
         cout<<"=== display data ==="<<endl; 
         for(size_t i=0;i< _table.size();i++) 
               cout<<"pos:"<<i<<" ,value:"<< _table[i]<<endl;
        }
// interpolation 
//--------linear interpolation ------//
double linearInterpolation(double ); // one variable
double linearInterpolation(double, double); // two variables
double linearInterpolation(double, double, double);  // three variables
double linearExtrapolation(double ); // one variable
double linearExtrapolation(double, double); // two variables
double linearExtrapolation(double, double, double);  // three variables
private:
valarray<double> _table;  
string _var1,_var2,_var3;
size_t _nb_pts1,_nb_pts2,_nb_pts3;
double _min1,_min2,_min3;
double _max1,_max2,_max3;
}; 
#endif // define _Table