// ------ Parameter.cpp ------
// S. CLAIN 2010/12
#include <Parameter.h>


/*----   open a PARAMETER balise and make the map -------*/
Parameter::Parameter(const char * filename)
{
string xml_files; 
StringMap attribute; 
StringMap::iterator iter;
SparseXML spxml(filename,xml_files);
size_t code,level;
// open  balise FVLIB 
 if(!_param.empty())
  {_param.erase(_param.begin(),_param.end());}
code=spxml.openBalise("FVLIB");
if (code!=OkOpenBalise)
       {cout<<" No open VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
// ok it is an FVLIB filename now find the PARAMETER  balise
code=spxml.openBalise("PARAMETER");
if (code!=OkOpenBalise)
       {cout<<" No open VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
level=spxml.getLevel();            
bool more_balise=true;
while(more_balise)
            {
              code=spxml.openBalise("parameter",level);
              if(code==NoOpenBalise) {more_balise=false; continue;}
              if(code==BadBaliseFormat) 
                    {
                     cout<<" BadBaliseFormat in parameter balise :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
                     exit(1) ;
                    } 
              if(code==EndXMLFile)  return ;    
              // ok, we have a Balise parameter
              attribute=spxml.getAttribute();              
              for( iter = attribute.begin(); iter !=attribute.end(); ++iter ) 
                     {
                       _param.insert( make_pair( iter->first ,iter->second ) );
                      // cout << "Key: '" << iter->first << "', Value: " << iter->second << endl;
                     }
            }
 // close  Balise   PARAMETER  
code=spxml.closeBalise("PARAMETER");
if (code!=OkCloseBalise)
       {cout<<" No close PARAMETER balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}     
// close  Balise   FVLIB   
code=spxml.closeBalise("FVLIB");
if (code!=OkCloseBalise)
       {cout<<" No close VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}
}










