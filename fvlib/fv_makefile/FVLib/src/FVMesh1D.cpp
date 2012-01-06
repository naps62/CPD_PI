// ------ FVMesh1D.cpp ------
// S. CLAIN 2011/07
#include <FVMesh1D.h>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include "Gmsh.h"
FVMesh1D::FVMesh1D()
{
    _nb_vertex=0;_nb_cell=0;_dim=1,_nb_boundary_vertex=0;
}    


FVMesh1D::FVMesh1D(const char *filename)
{
    FVMesh1D::read(filename);
}

size_t FVMesh1D::read(const char *filename)
{
    _spxml.readXML(filename,_xml); // read the file and put in the XML string
    string key, value,element;
    StringMap attribute;  
    //StringMap::iterator iter;
    size_t code;
           // open  balise FVLIB 
    if (_spxml.openBalise("FVLIB")!=OkOpenBalise)
            {cout<<" No open VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;return(NoOpenBalise);}    
 
//  find the FIELD  balise
    code=_spxml.openBalise("MESH");
    if (code!= OkOpenBalise)
        {cout<<" No open MESH balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);} 
    attribute=_spxml.getAttribute();           
    // read the dim of the mesh     
    key=string("dim");
    if(key.empty()) 
         {cout<<" No dim attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
    value= attribute[key];    
    _dim= (unsigned) atoi( value.c_str()); 
    if(_dim!=1)
        {
        #ifdef _DEBUGS
        cout<<" dimension do not match:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
        #endif      
        return(FVWRONGDIM);
        }   
    // read the name of the mesh     
    key=string("name");
    if(key.empty()) 
         {cout<<" No name attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);} 
    _name= attribute[key];
// ------------------------ we read the vertices  ------------------
    code=_spxml.openBalise("VERTEX");    
    if (code!= OkOpenBalise)
        {cout<<" No open VERTEX balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
    attribute=_spxml.getAttribute();     
        // read the number of vertices
     key=string("nbvertex");
     if(key.empty()) 
          {cout<<" No nbvertex attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
     value= attribute[key];
     _nb_vertex=(unsigned) atoi( value.c_str()); 
         // read the Vertices data
    _vertex.resize(_nb_vertex);
    _spxml.data();
    {
        size_t beginDATA=_spxml.getPosition();
        size_t lengthDATA=_spxml.getLength();
        char  thedata[lengthDATA+1],*ptr;
        _xml.copy(thedata,lengthDATA,beginDATA);
        thedata[lengthDATA]=0;
//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
        ptr=thedata;
        size_t count=0;
        while(count<_nb_vertex)// read the data and put it in the valarray
            {
            _vertex[count].label= strtod(ptr, &ptr);
            _vertex[count].code= strtod(ptr, &ptr);
            _vertex[count].coord.x= strtod(ptr, &ptr);
             count++; 
            }
    
     }
    // close  Balise   VERTEX
    code=_spxml.closeBalise("VERTEX");
    if (code!= OkCloseBalise)
        {cout<<" No close VERTEX balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}         
//----------------  we read the cells --------------------------
    code=_spxml.openBalise("CELL");    
    if (code!= OkOpenBalise)
        {cout<<" No open CELL balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}    
    attribute=_spxml.getAttribute();     
           // read the number of cell
     key=string("nbcell");
     if(key.empty()) 
          {cout<<" No nbcell attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
     value= attribute[key];
     _nb_cell=(unsigned) atoi( value.c_str());  
     
         // read the CELL data
    _cell.resize(_nb_cell);
    _spxml.data();
   
    {
        size_t beginDATA=_spxml.getPosition();
        size_t lengthDATA=_spxml.getLength();
        char  thedata[lengthDATA+1],*ptr;
        _xml.copy(thedata,lengthDATA,beginDATA);
        thedata[lengthDATA]=0;
//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
        ptr=thedata;
        size_t count=0,_label;
        
        while(count<_nb_cell)// read the data and put it in the valarray
            {
            _label= strtod(ptr, &ptr);
            _cell[_label-1].label=_label;
            _cell[_label-1].code= strtod(ptr, &ptr);
            _cell[_label-1].nb_vertex= strtod(ptr, &ptr);  // should be always equal to 2      
            _cell[_label-1].firstVertex= &(_vertex[strtod(ptr, &ptr)-1]);// label 1 correspond to vector[0]
            _cell[_label-1].secondVertex= &(_vertex[strtod(ptr, &ptr)-1]);         
             count++; 
            }
     }    
    // close  Balise   CELL
    code=_spxml.closeBalise("CELL");
    if (code!= OkCloseBalise)
        {cout<<" No close CELL balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}          
//---------------------------------------------                
    // close  Balise   MESH
    code=_spxml.closeBalise("MESH");
    if (code!= OkCloseBalise)
        {cout<<" No close MESH balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);} 
    // close  Balise   FVLIB
    code=_spxml.closeBalise("FVLIB");
    if (code!= OkCloseBalise)
        {cout<<" No close FVLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}  
// we have store all the data, now we complete the vertices and cells information        
   FVMesh1D::complete_data();  
return(FVOK);   
}

//  complete the vertices and cells information from the primitive data
void FVMesh1D::complete_data()
{
// we have store all the data, now we complete the vertices and cells information
for(size_t i=0;i<_nb_vertex;i++)
    {
    _vertex[i].leftCell=NULL;_vertex[i].rightCell=NULL;
    }  
for(size_t i=0;i<_nb_cell;i++)
    {
    _cell[i].centroid=(_cell[i].firstVertex->coord+_cell[i].secondVertex->coord)*0.5;
    _cell[i].length=_cell[i].firstVertex->coord.x-_cell[i].secondVertex->coord.x;
    if (_cell[i].length<0.) _cell[i].length*=-1.;
    if((_cell[i].firstVertex->coord.x-_cell[i].centroid.x)<0.) 
           _cell[i].first_normal.x=-1.;else _cell[i].first_normal.x=1.;
    _cell[i].second_normal=-_cell[i].first_normal;
    if(_cell[i].firstVertex->leftCell!=NULL) 
         _cell[i].firstVertex->rightCell=&(_cell[i]);
    else
         _cell[i].firstVertex->leftCell=&(_cell[i]);
    if(_cell[i].secondVertex->leftCell!=NULL) 
         _cell[i].secondVertex->rightCell=&(_cell[i]);
    else
         _cell[i].secondVertex->leftCell=&(_cell[i]);    
    }        
for(size_t i=0;i<_nb_vertex;i++)
    {
    if ((_vertex[i].coord.x-_vertex[i].leftCell->centroid.x)<0)   
         _vertex[i].normal.x=-1;
    else
         _vertex[i].normal.x=1;  
    if(!_vertex[i].rightCell) _boundary_vertex.push_back(&_vertex[i]);
    }       
_nb_boundary_vertex=_boundary_vertex.size();     
} 




size_t FVMesh1D::write(const char *filename)
{
if((_nb_cell==0) || (_nb_vertex==0))
            {
             #ifdef _DEBUGS  
             cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" there is no mesh to save"<<endl;
             #endif 
             return(FVERROR);
            }    
ofstream  out_file;    
out_file.open(filename);
if(!out_file.is_open())
            {
             #ifdef _DEBUGS   
             cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" can not create the file"<<endl;
             #endif 
             return(FVNOFILE);
            }
            
out_file<<"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"<<endl;
out_file<<"<FVLIB  >"<<endl; 
out_file<<"    <MESH dim=\"1\"    name=\""<<_name<<"\">"<<endl;
out_file<<"         <VERTEX nbvertex=\""<<_nb_vertex<<"\">"<<endl;
for(size_t i=0;i<_nb_vertex;i++)
    {
    out_file<< setw(FVCHAMPINT)<<_vertex[i].label<< setw(FVCHAMPINT)<< _vertex[i].code;  
    out_file<<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << _vertex[i].coord.x<<endl; 
    }
out_file<<"         </VERTEX>"<<endl; 
out_file<<"         <CELL nbcell=\""<<_nb_cell<<"\">"<<endl;
for(size_t i=0;i<_nb_cell;i++)
    {
    out_file<< setw(FVCHAMPINT)<<_cell[i].label<< setw(FVCHAMPINT)<< _cell[i].code
            << setw(FVCHAMPINT)<< _cell[i].nb_vertex; 
    out_file<<setw(FVCHAMPINT) << _cell[i].firstVertex->label<<setw(FVCHAMPINT) << _cell[i].secondVertex->label<<endl; 
    } 
out_file<<"         </CELL>"<<endl;      
out_file<<"    </MESH>"<<endl;
out_file<<"</FVLIB>"<<endl;          
out_file.close(); 
return(FVOK);
}






void FVMesh1D::Gmsh2FVMesh( Gmsh &m) // convert a Gmsh struct into a FVMesh1D
{
 if (m.getDim()!=1)
                 {
             cout<<" Dimension don't match. The Gmsh mesh should contain only 0 or 1D elements"<<endl;
             return;
            }  
_nb_vertex=m.getNbNode();
_name="mesh1D  from gmsh file"; 
_vertex.resize(_nb_vertex);
for(size_t i=0;i<_nb_vertex;i++)
    {
    _vertex[i].label=m.getNode(i)->label;
    _vertex[i].coord.x=m.getNode(i)->coord.x;
    _vertex[i].code=0;  // default value
    }
   // convention des code 
   // 0 est plus faible 2^29 est plus faible que 2^30 qui est plus faible que 2^31 qui est plus faible que pas de level  
_nb_cell=0;   
_cell.resize(0);
FVCell1D cell;
cell.length=0.;
// We first treat the cells
for(size_t i=0;i<m.getNbElement();i++)
    {  
    GMElement *ptr_el=m.getElement(i);  
    if(ptr_el->type_element!=1) continue; // we treat the node later  
    // take the minimum of information to detect cell identities
    cell.firstVertex=&(_vertex[ptr_el->node[0]-1]);
    cell.secondVertex=&(_vertex[ptr_el->node[1]-1]);
    cell.nb_vertex=2;
    bool exist=false;
    for(size_t j=0;j<_cell.size();j++)// look if the element still exist
         {
         if(isEqual(&_cell[j],&cell)) 
              {
               exist=true;
               if (ptr_el->code_physical>=MINUS_ONE_DIM)               
                   {_cell[j].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}
               else
                   _cell[j].code=ptr_el->code_physical;
               }
         }
    if(!exist)
        {
         cell.label=_cell.size()+1;
         if (ptr_el->code_physical>=MINUS_ONE_DIM)               
             {cell.setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM); cell.code=0;}
         else
             cell.code=ptr_el->code_physical;
        _cell.push_back(cell); // it is a real new cell so save it
        }
    }
_nb_cell=_cell.size();    
//we treat the node         
for(size_t i=0;i<m.getNbElement();i++)
    {  
    GMElement *ptr_el=m.getElement(i);  
    if(ptr_el->type_element!=15) continue;    
    _vertex[ptr_el->node[0]-1].code=ptr_el->code_physical;  // we set the Vertex code 
     } 
// complete the data
FVMesh1D::complete_data();     
}


