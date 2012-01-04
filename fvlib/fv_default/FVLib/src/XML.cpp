// ------ XML.cpp ------
// S. CLAIN 2010/12
#include <XML.h>



//-----  SparseXML section ----
SparseXML::SparseXML()
  {
_xml_string=NULL; 
_level=0;
_empty_element=true;
_current_position=0; // initialize the position 
_data_position=0;   
_data_length=0;
_element.clear();  // clean the element string
if(!_attribute.empty())  // clean the attribute map
              {_attribute.erase(_attribute.begin(),_attribute.end());}
  } 
//  
SparseXML::SparseXML(const char * filename,string &xml)
{
    SparseXML::readXML(filename,xml);
}
//
void SparseXML::readXML(const char * filename,string &xml)
{
_current_position=0; // initialize the position 
_data_position=0;   
_data_length=0;
_level=0;
_empty_element=true;
_element.clear();  // clean the element string
//
size_t filesize;
ifstream f(filename);
if(!f.is_open())
            {
             cout<<" error opening file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" file does not exist"<<endl;
             exit(0);
            }
filesize=f.tellg();
 _xml_string= &xml;
(*_xml_string).reserve(filesize);
f.seekg(0);
while (!f.eof())// read the file
      {
      (*_xml_string) += f.get(); 
      }
f.close();        
// first find the prologue       
    string begin_string_prologue="<?xml",end_string_prologue="?>";
    size_t begin_position_prologue=_xml_string->find(begin_string_prologue);
  if(begin_position_prologue==string::npos)
            {cout<<" error prologue begin format:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}  
    size_t end_position_prologue=_xml_string->find(end_string_prologue);
   if(end_position_prologue==string::npos)
            {cout<<" error prologue end format:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}   
   string prologue;
   begin_position_prologue+=5;// eliminate <?xml
   prologue.append((*_xml_string),begin_position_prologue,end_position_prologue-begin_position_prologue);
   end_position_prologue+=2; // eliminate  ?>
  // store the  attribute in the map.
   makeMap(prologue);
   // now we have to check if the is one root and  extract  
   _xml_string->erase(0,end_position_prologue);
   (*_xml_string)+='\0';
   // and we remove the comment balise
   removeComment();
   //cout<< "after the prologue"<<(*_xml_string) <<endl;
}
 // 
SparseXML::SparseXML(string &xml_string)
{
  cout<<"not built"<<endl;
 // I have got the string
} 
/*---------------------------- SPARSE FUNCTIONS -------------------*/
// remove all the comment balise
void SparseXML::removeComment()
{
string begin_string_comment="<!--",end_string_comment="-->";
size_t begin_position_comment,end_position_comment;
while((begin_position_comment=_xml_string->find(begin_string_comment))<string::npos)
       {// we have found the begining of a comment
        end_position_comment=_xml_string->find(end_string_comment);
        if(end_position_comment==string::npos)
              {cout<<" error comment without end "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);}  
         // now remove the comment 
         end_position_comment+=3; // to add -->
        _xml_string->erase(begin_position_comment,end_position_comment-begin_position_comment);
        }// end of the while   
} 
/* ------------ transform a string of attribute into a map   ------------*/
void SparseXML::makeMap(string &attribute_string)
{
  //cout<<"string to map: "<<attribute_string<<endl;
  // empty the map
  if(!_attribute.empty())
  {_attribute.erase(_attribute.begin(),_attribute.end());}
  // catch the attribute
  size_t pos=0;// initial position=0 in the string

  while(pos<attribute_string.length())
       {
        // go to the label 
        while (attribute_string.at(pos)==' ')
               {
                 pos++;
                 if(pos>=attribute_string.length()) return;// nothing more  
               }  
        // we have to read the label
        // we first clean the key and the value
        _key.clear();_value.clear();
        // we now search the key
        while(attribute_string.at(pos)!='=')
                {
                  _key+=attribute_string.at(pos);
                  pos++;
                  if(pos>=attribute_string.length())
                     {cout<<" can not find the key  in makeMap "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);} 
                }
         // pos is on the symbol =, find the first "
         while (attribute_string.at(pos)!='"')
                {
                  pos++;
                  if(pos>=attribute_string.length())
                     {cout<<" can not find quote in makeMap "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);} 
                }
          // we are on  the firt ", we skip it
         pos++;
         //we read the attribute      
         while (attribute_string.at(pos)!='"')
                  {
                  _value+=attribute_string.at(pos);                        
                  pos++;
                  if(pos>=attribute_string.length())
                     {cout<<" error reading attribute in makeMap "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; exit(1);} 
                } 
          // pos is on the second ", we go on
          pos++;
         //store (key value) into the map
         _attribute.insert( make_pair( _key, _value) );
        }
} 
/* -----------    open the next Balise ---------------------*/
int  SparseXML::openBalise()
{
// note that all the comment balise have been removed of the string xml  
//cout<<"openBalise"<<endl;
if(_current_position>=_xml_string->length())
      return(EndXMLFile);
// find the open chevron
while(_xml_string->at(_current_position)!='<')
          {
            _current_position++;
            if(_current_position>=_xml_string->length()) return EndXMLFile;
           }
// we have to check that we do not deal with a close balise
if(_xml_string->at(_current_position+1)=='/')
          {// it is a close balise
            _current_position--; //reposition of the pointer
            return(NoOpenBalise); // we can not open more balise
          } 
// ok it is an open balise
_level++;
_element.clear();
_current_position++; // skip the open chevron           
// skip the spaces
while (_xml_string->at(_current_position)==' ')
           {
            _current_position++;
            if(_current_position>=_xml_string->length())
                      {cout<<" error  function openBalise "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
                        return(BadBaliseFormat);}    
            }
//  _current_position is on the element 
while(_xml_string->at(_current_position)!=' ' &&
          _xml_string->at(_current_position)!='/'&& 
          _xml_string->at(_current_position)!='>')
            {
              _element+=_xml_string->at(_current_position);
              _current_position++;
              if(_current_position>=_xml_string->length())
                      {cout<<" error  function openBalise "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
                        return(BadBaliseFormat);}    
            }
// extract attribute line if any
string  attribute_string;
string end1=">";
string end2="/>";
size_t end_position_attribute1,end_position_attribute2;
attribute_string.clear();

end_position_attribute1=_xml_string->find(end1,_current_position);
end_position_attribute2=_xml_string->find(end2,_current_position);
if(end_position_attribute1==string::npos) // check that we close the chevron
              {cout<<" bad format for open balise  "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
                return(BadBaliseFormat);} 
if(end_position_attribute2<end_position_attribute1)   //\we have an empty element
      {
       // cout<<"it is an empty balise"<<endl;
       _empty_element=true; // it is an empty element
       attribute_string.append((*_xml_string),_current_position,end_position_attribute2-_current_position);
       _data_length=0;// no data
      _current_position= end_position_attribute2+2;// jump />
      }
else // there is possible data
      {
       // cout<<"balise with possible data"<<endl;
       _empty_element=false; 
        //cout<<"my position:"<<_xml_string->at(_current_position-2)<<_xml_string->at(_current_position-1)<<endl;       
        attribute_string.append((*_xml_string),_current_position,end_position_attribute1-_current_position);
        //cout<<"the attribute_string "<<attribute_string<<endl;
        _current_position= end_position_attribute1+1;// jump >
      }
makeMap(attribute_string);        
return(OkOpenBalise);     
}
/* -----------    open the  Balise  element ---------------------*/
int  SparseXML::openBalise(const char *name,size_t level)
{
string element=name;
return(openBalise(element,level));
}
//
int  SparseXML::openBalise(const string &element,size_t level)
{
size_t code;  

while(1)
      {
        code= openBalise();
        if(code==EndXMLFile) return(EndXMLFile);
        if(code==BadBaliseFormat) return(BadBaliseFormat);
        if(code==NoOpenBalise) 
              {
                code= closeBalise();
                if(_level<=level) return(NoOpenBalise);
              }  
        if(code==OkOpenBalise)
               {
                 if(element==_element) return(OkOpenBalise);
               }     
      }     
}


/*   ---------------  close the next balise  --------------------------*/
int  SparseXML::closeBalise()
{
//cout<<"closeBalise"<<endl;
if(_current_position>=_xml_string->length())
      return(EndXMLFile);
// find the open chevron
while(_xml_string->at(_current_position)!='<')
          {
            _current_position++;
            if(_current_position>=_xml_string->length())  return(EndXMLFile);
           }
// we have to check that we  deal with a close balise
if(_xml_string->at(_current_position+1)!='/')
          {// it is not  a close balise
            _current_position--; //reposition of the pointer
            return(NoCloseBalise); // we can not open more balise
          } 
// ok, we are sure to deal with a close balise, find the element
_level--;
_element.clear();
_current_position+=2; // skip the </           
// skip the spaces
while (_xml_string->at(_current_position)==' ')
           {
            _current_position++;
            if(_current_position>=_xml_string->length())
                      {cout<<" error  function closeBalise "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
                        return(BadBaliseFormat);}    
            }
// we get the element name            
while(_xml_string->at(_current_position)!=' ' &&
          _xml_string->at(_current_position)!='>')
            {
              _element+=_xml_string->at(_current_position);
              _current_position++;
              if(_current_position>=_xml_string->length())
                      {cout<<" error  function closeBalise "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
                        return(BadBaliseFormat);}
// new position after the close chevron >                        
            }            
  while(_xml_string->at(_current_position)!='>')
            {
            _current_position++;
            if(_current_position>=_xml_string->length())
                      {cout<<" error  function closeBalise "<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
                        return(BadBaliseFormat);}    
            }
_current_position++;   
// clean the map
string  attribute_string;
attribute_string.clear();
makeMap(attribute_string);  
return(OkCloseBalise);    
}  
/*   ---------------  close the  balise  element --------------------------*/
int  SparseXML::closeBalise(const char *name,size_t level)
{
string element=name;
return(closeBalise(element,level));
}
int  SparseXML::closeBalise(const string &element,size_t level)
{
size_t code;  

while(1)
      {
        code= closeBalise();
        if(code==EndXMLFile) return(EndXMLFile);
        if(code==BadBaliseFormat) return(BadBaliseFormat);
        if(code==NoCloseBalise) 
               {
                 code= openBalise();
                 if(_level<=level) return(NoCloseBalise);
               } 
        if(code==OkCloseBalise)
               {
                 if(element==_element) return(OkCloseBalise);
               }     
      }     
}
/*-----------   determine the data zone  ------------------------------*/
void  SparseXML::data()
{
  // the pointer _current_position is always outside of the <...> and considered as the begening of a data set
_data_position=_current_position;
  // find the first <
string chevron="<";
size_t end_data_position;
end_data_position=_xml_string->find(chevron,_current_position);
if(end_data_position==string::npos) 
     {_data_length=0;} // no data
else
     {_data_length=end_data_position-_current_position;}     
}