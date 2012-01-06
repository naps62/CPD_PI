 // ------ Table.cpp ------
// S. CLAIN 2010/12
#include <Table.h>

/*-------- read the function in file filename with name functionname ----*/ 
Table::Table(const char *filename,const char *functionname)
{
string xml_files; 
string key, value,element;
StringMap attribute;  
// initialization of the variable
_var1="";_var2="";_var3="";
_nb_pts1=0;_nb_pts2=0;_nb_pts3=0;
_min1=INF_MIN;_min2=INF_MIN;_min3=INF_MIN;
_max1=SUP_MAX;_max2=SUP_MAX;_max3=SUP_MAX;

StringMap::iterator iter;
SparseXML spxml(filename,xml_files);
size_t code;
//size_t level;
// open  balise FVLIB 

code=spxml.openBalise("FVLIB");
if (code!=OkOpenBalise)
       {cout<<" No open VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
//  find the TABLE  balise
code=spxml.openBalise("TABLE");
if (code!=OkOpenBalise)
       {cout<<" No open TABLE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
//level=
spxml.getLevel();       
// now we are in the Table find the function   functionname
while(1)
      {
        code=spxml.openBalise("FUNCTION");
        if(code!=OkOpenBalise)
              {cout<<" No FUNCTION balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}           
        attribute=spxml.getAttribute();
        key=string("label");
        value= attribute[key];
        if(value==string(functionname)) break; // ok we have found the right function
      }
// we have the right function
//
//
       // take the first VARIABLE
code=spxml.openBalise();
if(code!=OkOpenBalise)
              {cout<<" No VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}   
element=spxml.getElement();
if(element!=string("VARIABLE"))
            {cout<<" No VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
     // take the attribute
attribute= spxml.getAttribute();
key=string("label"); value=attribute[key];
if(key.empty()) 
       {cout<<" No label in VARIABLE 1 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_var1=value;  
key=string("nb_pts"); value=attribute[key];
if(key.empty()) 
       {cout<<" No nb_pts in VARIABLE 1 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_nb_pts1=(unsigned) atoi( value.c_str()) ;
key=string("min"); value=attribute[key];
if(key.empty()) 
       {cout<<" No min in VARIABLE 1 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_min1=atof( value.c_str()) ;
key=string("max"); value=attribute[key];
if(key.empty()) 
       {cout<<" No max in VARIABLE 1 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_max1=atof( value.c_str()) ;
//cout<<_var1<<" "<<_nb_pts1<<" "<<_min1<<" "<<_max1<<endl;
// take the second Balise  if exist or we have DATA
code=spxml.openBalise();
if(code!=OkOpenBalise)
              {cout<<" No VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}              
element=spxml.getElement();
if(element==string("DATA")) goto readDATA;// we have only on parameter go to read the DATA
if(element!=string("VARIABLE"))
            {cout<<" No DATA or VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
       // take the attribute
attribute= spxml.getAttribute();
key=string("label"); value=attribute[key];
if(key.empty()) 
       {cout<<" No label in VARIABLE 2 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_var2=value;  
key=string("nb_pts"); value=attribute[key];
if(key.empty()) 
       {cout<<" No nb_pts in VARIABLE 2 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_nb_pts2=(unsigned) atoi( value.c_str()) ;
key=string("min"); value=attribute[key];
if(key.empty()) 
       {cout<<" No min in VARIABLE 2 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_min2=atof( value.c_str()) ;
key=string("max"); value=attribute[key];
if(key.empty()) 
       {cout<<" No max in VARIABLE 2 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_max2=atof( value.c_str()) ;
//cout<<_var2<<" "<<_nb_pts2<<" "<<_min2<<" "<<_max2<<endl;     
 // take the third Balise  if exist or we have DATA
code=spxml.openBalise();
if(code!=OkOpenBalise)
              {cout<<" No VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}              
element=spxml.getElement();
if(element==string("DATA")) goto readDATA;// we have only on parameter go to read the DATA
if(element!=string("VARIABLE"))
            {cout<<" No DATA or VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
       // take the attribute
attribute= spxml.getAttribute();
key=string("label"); value=attribute[key];
if(key.empty()) 
       {cout<<" No label in VARIABLE 3 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_var3=value;  
key=string("nb_pts"); value=attribute[key];
if(key.empty()) 
       {cout<<" No nb_pts in VARIABLE 3 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_nb_pts3=(unsigned) atoi( value.c_str()) ;
key=string("min"); value=attribute[key];
if(key.empty()) 
       {cout<<" No min in VARIABLE 3 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_min3=atof( value.c_str()) ;
key=string("max"); value=attribute[key];
if(key.empty()) 
       {cout<<" No max in VARIABLE 3 balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
_max3=atof( value.c_str()) ;
//cout<<_var3<<" "<<_nb_pts3<<" "<<_min3<<" "<<_max3<<endl;                
 // now, we have to open DATA balise
code=spxml.openBalise();
if(code!=OkOpenBalise)
              {cout<<" No VARIABLE balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}              
element=spxml.getElement();
if(element!=string("DATA"))
            {cout<<" No DATA  balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
//  ok, we have all the parameter, now read the data
readDATA:   // section to read the data
spxml.data();
size_t beginDATA=spxml.getPosition();
size_t lengthDATA=spxml.getLength();
size_t pos=0;
size_t nb_data;
if((_nb_pts3==0)&&(_nb_pts2==0)) nb_data=_nb_pts1;
if((_nb_pts3==0)&&(_nb_pts2!=0))  nb_data=_nb_pts1*_nb_pts2;
if((_nb_pts3!=0)&&(_nb_pts2==0)) 
     {cout<<" Error in Data format  :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
if((_nb_pts3!=0)&&(_nb_pts2!=0))  nb_data=_nb_pts1*_nb_pts2*_nb_pts3;     
//cout<<"search "<<nb_data<<" values in DATA"<<endl;
//cout <<"position="<<beginDATA<<" length="<<lengthDATA<<endl;
size_t count=0;
_table.resize(nb_data);
while(count<nb_data)// read the data and put it in the valarray
        {
          while(xml_files.at(beginDATA+pos)<46 || xml_files.at(beginDATA+pos)>57)// skip the blank
                  {
                    pos++;
                    if(pos>lengthDATA)
                        {cout<<" Number of data inferior :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
                  }    
         // we are on the first unblank character
         value.clear();
          while(xml_files.at(beginDATA+pos)<=57 && xml_files.at(beginDATA+pos)>=46)// skip the blank
                  {                
                    value+=xml_files.at(beginDATA+pos);
                    pos++;
                    if(pos>lengthDATA)
                        {cout<<" Error reading DATA balise :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
                  }         
         _table[count]=atof(value.c_str());
         count++; 
        }
//
//
// close  Balise   TABLE
code=spxml.closeBalise("TABLE");
if (code!=OkCloseBalise)
       {cout<<" No close TABLE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}     
// close  Balise   FVLIB   
code=spxml.closeBalise("FVLIB");
if (code!=OkCloseBalise)
       {cout<<" No close VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
}
/*================ Interpolation section==================*/
//------ linear interpolation for one variable ----//
double Table::linearInterpolation(double x1)
{
  volatile size_t i1;
  volatile double coef1,OneOverDelta1,tab1m,tab1M;
  if(_nb_pts1<=1) 
      {cout<<"need at leat two elements"<<endl; exit(1);}
  if((x1<_min1) ||(x1>_max1)) 
    {cout<<"out of range"<<endl; exit(1);}      
  OneOverDelta1=(_nb_pts1-1)/(_max1-_min1);
  i1=floor( (x1-_min1)*OneOverDelta1);
  tab1m=_table[i1];tab1M=_table[i1+1];
  coef1=x1*OneOverDelta1-i1;
  return(tab1M*coef1+tab1m*(1.-coef1));

}
double Table::linearExtrapolation(double x1)
{
  volatile size_t i1;
  volatile double coef1,OneOverDelta1,tab1,tab2;
  if(_nb_pts1<=1) 
      {cout<<"need at leat two elements"<<endl; exit(1);}
  if(x1<_min1) return _table[0];
  if(x1>_max1) return _table[_nb_pts1-1];
  OneOverDelta1=(_nb_pts1-1)/(_max1-_min1);
  i1=floor((x1-_min1)*OneOverDelta1);
  tab1=_table[i1];tab2=_table[i1+1];
  coef1=x1*OneOverDelta1-i1;
  return(tab1*coef1+tab2*(1.-coef1));
}

//------ linear interpolation for two variables ----//
double Table::linearInterpolation(double x1, double x2)
{
  volatile size_t i1,i2;
  volatile double coef1,OneOverDelta1,tab11,tab12;  
  volatile double coef2,OneOverDelta2,tab21,tab22; 

  if(_nb_pts1<=1||_nb_pts2<=1 ) 
      {cout<<"need at leat two elements for each direction"<<endl; exit(1);}
  if(x1<_min1|| x1>_max1 || x2<_min2 || x2>_max2)
      {cout<<"out of range"<<endl; exit(1);}
 // compute the interpolation using barycentric coordinates 
  OneOverDelta1=(_nb_pts1-1)/(_max1-_min1);
  OneOverDelta2=(_nb_pts2-1)/(_max2-_min2);
  i1=floor((x1-_min1)*OneOverDelta1);
  i2=floor((x2-_min2)*OneOverDelta2); 
  tab11=_table[i1*_nb_pts2+i2];
  tab12=_table[i1*_nb_pts2+i2+1];
  tab21=_table[(i1+1)*_nb_pts2+i2];
  tab22=_table[(i1+1)*_nb_pts2+i2+1];  
  coef1=x1*OneOverDelta1-i1;
  coef2=x2*OneOverDelta2-i2;
  return(tab11+coef1*(tab12-tab11)+coef2*(tab21-tab11)+coef1*coef2*(tab22+tab11-tab12-tab21));
}
double Table::linearExtrapolation(double x1, double x2)
{
  volatile size_t i1,i2,inc1,inc2;
  volatile double coef1,OneOverDelta1,tab11,tab12;  
  volatile double coef2,OneOverDelta2,tab21,tab22;

  if(_nb_pts1<=1||_nb_pts2<=1 ) 
      {cout<<"need at leat two elements for each direction"<<endl; exit(1);}
  OneOverDelta1=(_nb_pts1-1)/(_max1-_min1);
  OneOverDelta2=(_nb_pts2-1)/(_max2-_min2);
  i1=floor((x1-_min1)*OneOverDelta1);
  i2=floor((x2-_min2)*OneOverDelta2);
  inc1=1;inc2=1;
  if(x1<_min1) {i1=0; inc1=0;}
  if(x1>_max1)  {i1=_nb_pts1-1,inc1=0;}
  if(x2<_min2)   {i2=0; inc2=0;}
  if(x2>_max2)  {i2=_nb_pts2-1,inc2=0;}
  // select the point to interpolate
  tab11=_table[i1*_nb_pts2+i2];
  tab12=_table[i1*_nb_pts2+i2+inc2];
  tab21=_table[(i1+inc1)*_nb_pts2+i2];
  tab22=_table[(i1+inc1)*_nb_pts2+i2+inc2];  
  // compute the interpolation using barycentric coordinates 
  coef1=x1*OneOverDelta1-i1;
  coef2=x2*OneOverDelta2-i2;
  return(tab11+coef1*(tab12-tab11)+coef2*(tab21-tab11)+coef1*coef2*(tab22+tab11-tab12-tab21)); 
}
//------ linear interpolation for three variables ----//
double Table::linearInterpolation(double x1, double x2, double x3)
{
  volatile size_t i1,i2,i3;
  volatile double coef1,OneOverDelta1;
  volatile double coef2,OneOverDelta2; 
  volatile double coef3,OneOverDelta3; 
  volatile double tab111,tab112,tab121,tab122,tab211,tab212,tab221,tab222;
  
  
  if(_nb_pts1<=1||_nb_pts2<=1||_nb_pts3<=1 ) 
      {cout<<"need at leat two elements for each direction"<<endl; exit(1);}
  if(x1<_min1|| x1>_max1 || x2<_min2 || x2>_max2 || x3<_min3 || x3>_max3)
      {cout<<"out of range"<<endl; exit(1);}
 // compute the interpolation using barycentric coordinates 
  OneOverDelta1=(_nb_pts1-1)/(_max1-_min1);
  OneOverDelta2=(_nb_pts2-1)/(_max2-_min2);
  OneOverDelta3=(_nb_pts3-1)/(_max3-_min3);  
  i1=floor((x1-_min1)*OneOverDelta1);
  i2=floor((x2-_min2)*OneOverDelta2); 
  i3=floor((x3-_min3)*OneOverDelta3);  
   
  tab111=_table[(i1*_nb_pts2+i2)*_nb_pts3+i3];
  tab112=_table[(i1*_nb_pts2+i2)*_nb_pts3+i3+1];
  tab121=_table[(i1*_nb_pts2+i2+1)*_nb_pts3+i3];
  tab122=_table[(i1*_nb_pts2+i2+1)*_nb_pts3+i3+1]; 
  tab211=_table[((i1+1)*_nb_pts2+i2)*_nb_pts3+i3];
  tab212=_table[((i1+1)*_nb_pts2+i2)*_nb_pts3+i3+1]; 
  tab221=_table[((i1+1)*_nb_pts2+i2+1)*_nb_pts3+i3];
  tab222=_table[((i1+1)*_nb_pts2+i2+1)*_nb_pts3+i3+1];   
  coef1=x1*OneOverDelta1-i1;
  coef2=x2*OneOverDelta2-i2;
  coef3=x3*OneOverDelta3-i3; 
 volatile double alpha=tab111+coef1*(tab211-tab111)+coef2*(tab121-tab111)+coef3*(tab112-tab111);
 alpha+=coef1*coef2*(tab111-tab211-tab121+tab221);
 alpha+=coef1*coef3*(tab111-tab211-tab112+tab212);
 alpha+=coef2*coef3*(tab111-tab121-tab112+tab122);
 alpha+=coef1*coef2*coef3*(tab222+tab112+tab121+tab211-tab111-tab221-tab212-tab122);
 return(alpha);
}
double Table::linearExtrapolation(double x1, double x2,double x3)
{
  volatile size_t i1,i2,i3,inc1,inc2,inc3;
  volatile double coef1,OneOverDelta1;
  volatile double coef2,OneOverDelta2; 
  volatile double coef3,OneOverDelta3; 
  volatile double tab111,tab112,tab121,tab122,tab211,tab212,tab221,tab222;
 
  if(_nb_pts1<=1||_nb_pts2<=1||_nb_pts3<=1 ) 
      {cout<<"need at leat two elements for each direction"<<endl; exit(1);}
  if(x1<_min1|| x1>_max1 || x2<_min2 || x2>_max2 || x3<_min3 || x3>_max3)
      {cout<<"out of range"<<endl; exit(1);}
 // compute the interpolation using barycentric coordinates 
  OneOverDelta1=(_nb_pts1-1)/(_max1-_min1);
  OneOverDelta2=(_nb_pts2-1)/(_max2-_min2);
  OneOverDelta3=(_nb_pts3-1)/(_max3-_min3);  
  i1=floor((x1-_min1)*OneOverDelta1);
  i2=floor((x2-_min2)*OneOverDelta2); 
  i3=floor((x3-_min3)*OneOverDelta3);
  inc1=1;inc2=1;inc3=1;
  if(x1<_min1) {i1=0; inc1=0;}
  if(x1>_max1)  {i1=_nb_pts1-1,inc1=0;}
  if(x2<_min2)   {i2=0; inc2=0;}
  if(x2>_max2)  {i2=_nb_pts2-1,inc2=0;}  
  if(x3<_min3)   {i3=0; inc3=0;}
  if(x3>_max3)  {i3=_nb_pts3-1,inc3=0;}    
  tab111=_table[(i1*_nb_pts2+i2)*_nb_pts3+i3];
  tab112=_table[(i1*_nb_pts2+i2)*_nb_pts3+i3+inc3];
  tab121=_table[(i1*_nb_pts2+i2+inc2)*_nb_pts3+i3];
  tab122=_table[(i1*_nb_pts2+i2+inc2)*_nb_pts3+i3+inc3]; 
  tab211=_table[((i1+inc1)*_nb_pts2+i2)*_nb_pts3+i3];
  tab212=_table[((i1+inc1)*_nb_pts2+i2)*_nb_pts3+i3+inc3]; 
  tab221=_table[((i1+inc1)*_nb_pts2+i2+inc2)*_nb_pts3+i3];
  tab222=_table[((i1+inc1)*_nb_pts2+i2+inc2)*_nb_pts3+i3+inc3];   
  coef1=x1*OneOverDelta1-i1;
  coef2=x2*OneOverDelta2-i2;
  coef3=x3*OneOverDelta3-i3; 
 volatile double alpha=tab111+coef1*(tab211-tab111)+coef2*(tab121-tab111)+coef3*(tab112-tab111);
 alpha+=coef1*coef2*(tab111-tab211-tab121+tab221);
 alpha+=coef1*coef3*(tab111-tab211-tab112+tab212);
 alpha+=coef2*coef3*(tab111-tab121-tab112+tab122);
 alpha+=coef1*coef2*coef3*(tab222+tab112+tab121+tab211-tab111-tab221-tab212-tab122);
 return(alpha);
}
