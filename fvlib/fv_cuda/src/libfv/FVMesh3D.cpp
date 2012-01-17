#include <iostream>
#include <cstdio>
#include <math.h>
#include "FVMesh3D.h"
#include "Gmsh.h"
FVMesh3D::FVMesh3D()
{
    _nb_vertex=0;_nb_cell=0;_nb_edge=0,_nb_face=0;_dim=0;
    
}  

FVMesh3D::FVMesh3D(const char *filename)
{
    FVMesh3D::read(filename);
}


size_t FVMesh3D::read(const char *filename)
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
    if (code!=OkOpenBalise)
        {cout<<" No open MESH balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);} 
    attribute=_spxml.getAttribute();           
    // read the name of the mesh  
    key=string("dim");
    if(key.empty()) 
         {cout<<" No dim attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
    value= attribute[key];    
    _dim= (unsigned) atoi( value.c_str()); 
    if(_dim!=3)
         {
         #ifdef _DEBUGS   
         cout<<" dimension do not match:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
         #endif
         return(FVWRONGDIM);
         } 
    key=string("name");
    if(key.empty()) 
         {cout<<" No name attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);} 
    _name= attribute[key];

    
// ------------------------ we read the vertices  ------------------
    code=_spxml.openBalise("VERTEX");    
    if (code!=OkOpenBalise)
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
        char  *thedata,*ptr;
        thedata=(char *) malloc(lengthDATA+1);
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
            _vertex[count].coord.y= strtod(ptr, &ptr);  
            _vertex[count].coord.z= strtod(ptr, &ptr);          
             count++; 
            }
        free(thedata);
     }      
   
     // close  Balise   VERTEX
    code=_spxml.closeBalise("VERTEX");
    if (code!=OkCloseBalise)
        {cout<<" No close VERTEX balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}   
        
//----------------  we read the edges --------------------------        
       
     code=_spxml.openBalise("EDGE");    
    if (code!=OkOpenBalise)
        {cout<<" No open EDGE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
    attribute=_spxml.getAttribute();     
           // read the number of cell
     key=string("nbedge");
     if(key.empty()) 
          {cout<<" No nbedge attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
     value= attribute[key];
     _nb_edge=(unsigned) atoi( value.c_str());  
       
        // read the EDGE data
    _edge.resize(_nb_edge);
    _spxml.data();
   
    {
        size_t beginDATA=_spxml.getPosition();
        size_t lengthDATA=_spxml.getLength();
        char  *thedata,*ptr;
        thedata=(char *) malloc(lengthDATA+1);
        _xml.copy(thedata,lengthDATA,beginDATA);
        thedata[lengthDATA]=0;
//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
        ptr=thedata;
        size_t count=0,_label;
        
        while(count<_nb_edge)// read the data and put it in the valarray
            {
            _label= strtod(ptr, &ptr);
            _edge[_label-1].label=_label;
            _edge[_label-1].code= strtod(ptr, &ptr); 
            _edge[_label-1].nb_vertex= strtod(ptr, &ptr);   // should be always 2
            _edge[_label-1].firstVertex= &(_vertex[strtod(ptr, &ptr)-1]); 
            _edge[_label-1].secondVertex=&(_vertex[strtod(ptr, &ptr)-1]);     
             count++; 
            }
        free(thedata) ;   
     }           
       
    // close  Balise   EDGE
    code=_spxml.closeBalise("EDGE");
    if (code!=OkCloseBalise)
        {cout<<" No close EDGE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}         
     
     //----------------  we read the faces --------------------------        
       
     code=_spxml.openBalise("FACE");    
    if (code!=OkOpenBalise)
        {cout<<" No open FACE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
    attribute=_spxml.getAttribute();     
           // read the number of cell
     key=string("nbface");
     if(key.empty()) 
          {cout<<" No nbface attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
     value= attribute[key];
     _nb_face=(unsigned) atoi( value.c_str());  
         // read the EDGE data
    _face.resize(_nb_face);
    _spxml.data();
   
    {
        size_t beginDATA=_spxml.getPosition();
        size_t lengthDATA=_spxml.getLength();
        char  *thedata,*ptr;
        thedata=(char *) malloc(lengthDATA+1);
        _xml.copy(thedata,lengthDATA,beginDATA);
        thedata[lengthDATA]=0;
//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
        ptr=thedata;
        size_t count=0,_label;
        
        while(count<_nb_face)// read the data and put it in the valarray
            {
            _label= strtod(ptr, &ptr);
            _face[_label-1].label=_label;
            _face[_label-1].code= strtod(ptr, &ptr);
            _face[_label-1].nb_edge= strtod(ptr, &ptr);  
            for(size_t i=0;i<_face[_label-1].nb_edge;i++)
                       _face[_label-1].edge[i]= &(_edge[strtod(ptr, &ptr)-1]);        
            count++; 
            }
       free(thedata);
     }         
        
       
    // close  Balise   FACE
    code=_spxml.closeBalise("FACE");
    if (code!=OkCloseBalise)
        {cout<<" No close FACEbalise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}  
        
//----------------  we read the cells --------------------------

    code=_spxml.openBalise("CELL");    
    if (code!=OkOpenBalise)
        {cout<<" No open CELL balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
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
        char  *thedata,*ptr;
        thedata=(char *) malloc(lengthDATA+1);
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
            _cell[_label-1].nb_face= strtod(ptr, &ptr);  
            for(size_t i=0;i<_cell[_label-1].nb_face;i++)
                       _cell[_label-1].face[i]= &(_face[strtod(ptr, &ptr)-1]);        
            count++; 
            }
       free(thedata);      
     }  
        
    // close  Balise   CELL
    code=_spxml.closeBalise("CELL");
    if (code!=OkCloseBalise)
        {cout<<" No close CELL balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}          
//---------------------------------------------                
    // close  Balise   MESH
    code=_spxml.closeBalise("MESH");
    if (code!=OkCloseBalise)
        {cout<<" No close MESH balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);} 
    // close  Balise   FVLIB
    code=_spxml.closeBalise("FVLIB");
    if (code!=OkCloseBalise)
        {cout<<" No close FVLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}        
    //------- we have store all the data, now we complete the vertices and cells information 
 
    FVMesh3D::complete_data();
return(FVOK);
}    

void FVMesh3D::complete_data()
{

// initialize the list of cell for each vertex    
for(size_t i=0;i<_nb_vertex;i++)
    {
    _vertex[i].nb_cell=0;
    }  
// we have to determine the list of neighbor cell for each vertex further
// compute the centroid and length of edge
for(size_t i=0;i<_nb_edge;i++)
    {
    _edge[i].centroid=(_edge[i].firstVertex->coord+_edge[i].secondVertex->coord)*(fv_float)0.5;
    FVPoint3D<fv_float> u;
    u=_edge[i].firstVertex->coord-_edge[i].secondVertex->coord;
    _edge[i].length=Norm(u);
    }
// We have completly finish with the edge

// compute geometric stuff for the face
for(size_t i=0;i<_nb_face;i++)
    {
    _face[i].perimeter=0.;
    _face[i].area=0.;
    _face[i].centroid=0.;
    // conpute  perimeter or the face 
    for(size_t j=0;j<_face[i].nb_edge;j++)
        {
        _face[i].perimeter+=_face[i].edge[j]->length; 
        _face[i].centroid+=_face[i].edge[j]->centroid*_face[i].edge[j]->length; 
        }
    // compute the centroid of the face
    _face[i].centroid/=_face[i].perimeter;  
    // compute the area of the face
    for(size_t j=0;j<_face[i].nb_edge;j++)
        {    
        FVPoint3D<fv_float> u,v,w;  
        u=_face[i].edge[j]->firstVertex->coord-_face[i].centroid; 
        v=_face[i].edge[j]->secondVertex->coord-_face[i].centroid;
        w=CrossProduct(u,v);
        _face[i].area+=Norm(w)*0.5;
        }  
   // build the list of vertex pointer for the face  
    pos_v=0;
    for(size_t j=0;j<_face[i].nb_edge;j++)
        {
        bool _still_exist;   
        _still_exist=false; 
        for(size_t k=0;k<pos_v;k++)
             if(_face[i].edge[j]->firstVertex==_face[i].vertex[k])  _still_exist=true;
        if(!_still_exist) {_face[i].vertex[pos_v]=_face[i].edge[j]->firstVertex;pos_v++;}
        _still_exist=false;  
        for(size_t k=0;k<pos_v;k++)
             if(_face[i].edge[j]->secondVertex==_face[i].vertex[k])  _still_exist=true;
        if(!_still_exist) {_face[i].vertex[pos_v]=_face[i].edge[j]->secondVertex;pos_v++;}    
        }
    _face[i].nb_vertex=pos_v;         
    }
    //  left and right cell, normal vector will be determined after the loop on cells
// end loop on the faces  
for(size_t i=0;i<_nb_cell;i++)
    {
    _cell[i].surface=0.;
    _cell[i].volume=0.;
    _cell[i].centroid=0.;
    // conpute surface of the cell 
    // determine the left and right cell for the face
    for(size_t j=0;j<_cell[i].nb_face;j++)
        {
        size_t pos;    
        _cell[i].surface+=_cell[i].face[j]->area; 
        _cell[i].centroid+=_cell[i].face[j]->centroid*_cell[i].face[j]->area; 
        pos=_cell[i].face[j]->label-1;
        if(!(_face[pos].leftCell)  )  
             _face[pos].leftCell=&(_cell[i]);
        else
             _face[pos].rightCell=&(_cell[i]); 
        }
    // compute the centroid of the cell
    _cell[i].centroid/=_cell[i].surface;  
    // compute the cell2face vector
    // compute the volume of the cell
    for(size_t j=0;j<_cell[i].nb_face;j++)
        {
        _cell[i].cell2face[j]= _cell[i].face[j]->centroid-_cell[i].centroid;   
        for(size_t k=0;k<_cell[i].face[j]->nb_edge;k++)
            {
            FVPoint3D<fv_float> u,v,w;
            u=_cell[i].cell2face[j];
            v=_cell[i].face[j]->edge[k]->firstVertex->coord-_cell[i].centroid;
            w=_cell[i].face[j]->edge[k]->secondVertex->coord-_cell[i].centroid;   
            _cell[i].volume+=fabs(Det(u,v,w))/6;
            }
        }   
    // build the list of the vertex pointer for a cell     
    pos_v=0;
    for(size_t j=0;j<_cell[i].nb_face;j++)
        for(size_t k=0;k<_cell[i].face[j]->nb_edge;k++)
            {
            bool _still_exist;   
            _still_exist=false;  
            for(size_t m=0;m<pos_v;m++)
                 if(_cell[i].face[j]->edge[k]->firstVertex==_cell[i].vertex[m])  _still_exist=true;
            if(!_still_exist) {_cell[i].vertex[pos_v]=_cell[i].face[j]->edge[k]->firstVertex;pos_v++;}
            _still_exist=false;  
            for(size_t m=0;m<pos_v;m++)
                 if(_cell[i].face[j]->edge[k]->secondVertex==_cell[i].vertex[m])  _still_exist=true;
            if(!_still_exist) {_cell[i].vertex[pos_v]=_cell[i].face[j]->edge[k]->secondVertex;pos_v++;}  
            }
     _cell[i].nb_vertex=pos_v;  
    // build the list of the cell pointer for a vertex        
     for(size_t j=0;j<_cell[i].nb_vertex;j++)
        {
        size_t pos;
        pos=_cell[i].vertex[j]->label-1;
        _vertex[pos].cell[_vertex[pos].nb_cell]=&(_cell[i]); 
        _vertex[pos].nb_cell++;
       if(_vertex[pos].nb_cell>=NB_CELL_PER_VERTEX_3D)
         cout<<"Warning, overflow in class FVVertex3D, too many Cells, found "<<_vertex[pos].nb_cell<<endl; 
        }
    }   
//  we compute the normal from left to rigth for each sub-triangle   
_boundary_face.resize(0);
_nb_boundary_face=0;
for(size_t i=0;i<_nb_face;i++)
    {
    for(size_t j=0;j<_face[i].nb_edge;j++)
        {
         FVPoint3D<fv_float> u,v,w;  
         fv_float no;
         u=_face[i].edge[j]->firstVertex->coord-_face[i].centroid;
         v=_face[i].edge[j]->secondVertex->coord-_face[i].centroid;       
         w=CrossProduct(u,v);
         no=Norm(w);
         w/=no;
         _face[i].normal[j]=w;
         u=_face[i].centroid-_face[i].leftCell->centroid;
         if(w*u<0) _face[i].normal[j]*=-1.; 
         }     // build the list of boundary face
    if(! (_face[i].rightCell)) {_boundary_face.push_back(&(_face[i]));_nb_boundary_face++;} 
    }
}



size_t FVMesh3D::write(const char *filename)
{
if((_nb_cell==0) || (_nb_face==0) || (_nb_edge==0) || (_nb_vertex==0))
            {
             cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" there is no mesh to save"<<endl;
             return(FVERROR);
            }    
ofstream  out_file;    
out_file.open(filename);
if(!out_file.is_open())
            {
             cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" can not create the file"<<endl;
             return(FVNOFILE);
            }
            
out_file<<"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"<<endl;
out_file<<"<FVLIB>"<<endl; 
out_file<<"    <MESH dim=\"3\"    name=\""<<_name<<"\">"<<endl;
out_file<<"         <VERTEX nbvertex=\""<<_nb_vertex<<"\">"<<endl;
for(size_t i=0;i<_nb_vertex;i++)
    {
    out_file<< setw(FVCHAMPINT)<<_vertex[i].label<< setw(FVCHAMPINT)<< _vertex[i].code;  
    out_file<<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << _vertex[i].coord.x;
    out_file<<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << _vertex[i].coord.y;
    out_file<<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << _vertex[i].coord.z<<endl; 
    }
out_file<<"         </VERTEX>"<<endl; 
out_file<<"         <EDGE nbedge=\""<<_nb_edge<<"\">"<<endl;
for(size_t i=0;i<_nb_edge;i++)
    {
    out_file<< setw(FVCHAMPINT)<<_edge[i].label<< setw(FVCHAMPINT)<< _edge[i].code
            << setw(FVCHAMPINT)<< _edge[i].nb_vertex; 
    out_file<<setw(FVCHAMPINT) << _edge[i].firstVertex->label<<setw(FVCHAMPINT) << _edge[i].secondVertex->label<<endl; 
    } 
out_file<<"         </EDGE>"<<endl; 
out_file<<"         <FACE nbface=\""<<_nb_face<<"\">"<<endl;
for(size_t i=0;i<_nb_face;i++)
    {
    out_file<< setw(FVCHAMPINT)<<_face[i].label<< setw(FVCHAMPINT)<< _face[i].code
            << setw(FVCHAMPINT)<< _face[i].nb_edge;
    for(size_t j=0;j<_face[i].nb_edge;j++)        
             out_file<<setw(FVCHAMPINT) << _face[i].edge[j]->label<<setw(FVCHAMPINT); 
    out_file<<endl; 
    } 
out_file<<"         </FACE>"<<endl; 
out_file<<"         <CELL nbcell=\""<<_nb_cell<<"\">"<<endl;
for(size_t i=0;i<_nb_cell;i++)
    {
    out_file<< setw(FVCHAMPINT)<<_cell[i].label<< setw(FVCHAMPINT)<< _cell[i].code
            << setw(FVCHAMPINT)<< _cell[i].nb_face;
    for(size_t j=0;j<_cell[i].nb_face;j++)        
             out_file<<setw(FVCHAMPINT) << _cell[i].face[j]->label<<setw(FVCHAMPINT); 
    out_file<<endl; 
    } 
out_file<<"         </CELL>"<<endl;      
out_file<<"    </MESH>"<<endl;
out_file<<"</FVLIB>"<<endl;          
out_file.close();    
return(FVOK);
}





void FVMesh3D::Gmsh2FVMesh( Gmsh &m)  // convert a Gmsh struct into a FVMesh3D    
{
if (m.getDim()!=3)
                 {
             cout<<" Dimension don't match. The Gmsh mesh should contain only 0 or 1D elements"<<endl;
             return;
            }  
_nb_vertex=m.getNbNode();
_name="mesh3D  from gmsh file"; 
_vertex.resize(_nb_vertex); 
for(size_t i=0;i<_nb_vertex;i++)
    {
    _vertex[i].label=m.getNode(i)->label;
    _vertex[i].coord.x=m.getNode(i)->coord.x;
    _vertex[i].coord.y=m.getNode(i)->coord.y;
    _vertex[i].coord.z=m.getNode(i)->coord.z;
    _vertex[i].code=0;  // default value
    }     
_nb_cell=0;_nb_edge=0;_nb_face=0;   
_cell.resize(0);_face.resize(0);_edge.resize(0);
_cell.reserve(m.getNbElement());_face.reserve(m.getNbElement()*3);_edge.reserve(m.getNbElement()*5);
FVCell3D cell;
FVFace3D face;
FVEdge3D edge;
size_t pos_dim3=0;
bool exist;


for(size_t i=0;i<m.getNbElement();i++)
    {  
    GMElement *ptr_el=m.getElement(i);  
    if  (ptr_el->type_element==15) 
         {
          pos_dim3++;
          ptr_el->label=_vertex[ptr_el->node[0]-1].label;// label is no longer GMElement label but  give 
                                                         // the label of the vertex in the _vertex table
         }
    // 1D element
    if  (ptr_el->type_element==1)
         { 
         pos_dim3++;
         edge.firstVertex=&(_vertex[ptr_el->node[0]-1]);
         edge.secondVertex=&(_vertex[ptr_el->node[1]-1]);
         edge.nb_vertex=2;
         exist=false;
         for(size_t j=0;j<_edge.size();j++)// look if the edge still exist
             if(isEqual(&_edge[j],&edge)) {exist=true;ptr_el->label=j+1;}
         if(!exist)
             {
              ptr_el->label=edge.label=_edge.size()+1;    // label is no longer GMElement label but  give
                                                          // the label of the edge in the _edge table            
              edge.code=0;  // we see the code later
              _edge.push_back(edge); // it is a real new edge so save it
             }    
         }
         
         
         
         
         
    // 2D element         
    if  (ptr_el->type_element==2 ||ptr_el->type_element==3 )  // treat   a triangle or a quadrangle
         {
         //cout<<"creating cell"<<endl;fflush(NULL);  
         pos_dim3++;  
         // first we create the edges
         for(size_t j=0;j<=ptr_el->type_element;j++)
             {
              edge.firstVertex=&(_vertex[ptr_el->node[j]-1]);
              edge.secondVertex=&(_vertex[ptr_el->node[(j+1)%(ptr_el->type_element+1)]-1]);
              edge.nb_vertex=2;
              exist=false;
              for(size_t k=0;k<_edge.size();k++)// look if the edge still exist
                     if(isEqual(&_edge[k],&edge))  
                        {
                         face.edge[j]=&_edge[k];exist=true;                    
                        } // if exist we keep the pointer
              if(!exist) // if not exist create it
                  {  
                   edge.label=_edge.size()+1;             
                   edge.code=0;  // we see the code later
                   _edge.push_back(edge); // it is a real new edge so save it
                   face.edge[j]=&_edge[_edge.size()-1]; // and keep the pointer
                   }
               }
         // at that stage we have all the edge 
         // we creat the face if necessary and set the code by dimension reduction
         face.nb_edge=ptr_el->type_element+1;  
         exist=false;
         for(size_t j=0;j<_face.size();j++)// look if the face still exist
             if(isEqual(&_face[j],&face))  {exist=true; ptr_el->label=j+1;}// the face still exist,  
         if(!exist) 
             {
              //cout<<"the cell (gmsh label)"<<ptr_el->label<< " does not exist"<<endl;  fflush(NULL);
              //cout<<"the edge are "<<cell.edge[0]->label<<" "<<cell.edge[1]->label<<" "<<cell.edge[2]->label<<" "<<endl;
              ptr_el->label=face.label=_face.size()+1;  
              face.code=0;  // we see the code later
              _face.push_back(face); // it is a real new edge so save it

             }  
         }  
         
         
         
         
         
//         
//  now we treat the 3D elements         
//
    // treat  tetrahedron 
    if  (ptr_el->type_element==4) 
         {
          for(size_t j=0;j<4;j++)  //loop on the face
             {
              FVFace3D face;     // construct a Face of triangle
              for(size_t k=0;k<3;k++)  // loop on the edge
                  {
                  FVEdge3D edge;  //seek the Edge
                  size_t n1,n2,nd[3];
                  if(j==0) {nd[0]=1;nd[1]=2;nd[2]=3;}
                  if(j==1) {nd[0]=0;nd[1]=2;nd[2]=3;}      
                  if(j==2) {nd[0]=0;nd[1]=1;nd[2]=3;}   
                  if(j==3) {nd[0]=0;nd[1]=1;nd[2]=2;}                    
                  n1=nd[k];n2=nd[(k+1)%3]; 
                  edge.firstVertex=&(_vertex[ptr_el->node[n1]-1]);
                  edge.secondVertex=&(_vertex[ptr_el->node[n2]-1]);
                  edge.nb_vertex=2;             
                  exist=false;
                  for(size_t m=0;m<_edge.size();m++)// look if the edge still exist
                        {
                         if(isEqual(&_edge[m],&edge))  
                            { 
                             face.edge[k]=&_edge[m];exist=true;                    
                            } // if exist we keep the pointer
                        }         
                  if(!exist) // if not exist create it
                      {  
                       edge.label=_edge.size()+1;        
                       edge.code=0;  // we see the code later
                       _edge.push_back(edge); // it is a real new edge so save it
                       face.edge[k]=&_edge[_edge.size()-1]; // and keep the pointer
                       }                     
                   }
              // at that stage, the face has all its edge
              // check if the face still exists
              exist=false;
              face.nb_edge=3;             
              for(size_t m=0;m<_face.size();m++)// look if the face still exist
              if(isEqual(&_face[m],&face))  
                            {
                             cell.face[j]=&_face[m];exist=true;                    
                            } // if exist we keep the pointer
              if(!exist) // if not exist create it
                   {  
                   face.label=_face.size()+1;        
                   face.code=0;  // we see the code later
                   _face.push_back(face); // it is a real new face so save it
                   cell.face[j]=&_face[_face.size()-1]; // and keep the pointer
                    }  
               }    
         // at that stage we have all the face and edge now chesk the code
         cell.nb_face=4;   
         exist=false;
         for(size_t j=0;j<_cell.size();j++)// look if the cell still exist
             {    
             if(isEqual(&_cell[j],&cell))                 
                 {
                  exist=true;
                  if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {_cell[j].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}                  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {_cell[j].setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
                  if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {_cell[j].setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       _cell[j].code=ptr_el->code_physical;
                  }
             }      
         if(!exist) 
             {
              //cout<<"the cell (gmsh label)"<<ptr_el->label<< " does not exist"<<endl;  fflush(NULL);
              //cout<<"the edge are "<<cell.edge[0]->label<<" "<<cell.edge[1]->label<<" "<<cell.edge[2]->label<<" "<<endl;
              
              ptr_el->label=cell.label=_cell.size()+1;       
              if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {cell.setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}              
              if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {cell.setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
              if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {cell.setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}   
              if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       cell.code=ptr_el->code_physical;
              _cell.push_back(cell); // it is a real new cell so save it

             }  
         }
    // treat  hexahedron 
    if  (ptr_el->type_element==5) 
         {
          for(size_t j=0;j<6;j++)  //loop on the face
             {
              FVFace3D face;     // construct a Face of triangle
              for(size_t k=0;k<4;k++)  // loop on the edge
                  {
                  FVEdge3D edge;  //seek the Edge
                  size_t n1,n2,nd[4];
                  if(j==0) {nd[0]=0;nd[1]=1;nd[2]=2;nd[3]=3;}
                  if(j==1) {nd[0]=4;nd[1]=5;nd[2]=6;nd[3]=7;}
                  if(j==2) {nd[0]=0;nd[1]=3;nd[2]=7;nd[3]=4;}
                  if(j==3) {nd[0]=1;nd[1]=2;nd[2]=6;nd[3]=5;}                  
                  if(j==4) {nd[0]=2;nd[1]=3;nd[2]=7;nd[3]=6;}
                  if(j==5) {nd[0]=0;nd[1]=1;nd[2]=5;nd[3]=4;}                  
                  n1=nd[k];n2=nd[(k+1)%4];
                  edge.firstVertex=&(_vertex[ptr_el->node[n1]-1]);
                  edge.secondVertex=&(_vertex[ptr_el->node[n2]-1]);
                  edge.nb_vertex=2;             
                  exist=false;
                  for(size_t m=0;m<_edge.size();m++)// look if the edge still exist
                         if(isEqual(&_edge[m],&edge))  
                            {
                             face.edge[k]=&_edge[m];exist=true;                    
                            } // if exist we keep the pointer
                  if(!exist) // if not exist create it
                      {  
                       edge.label=_edge.size()+1;      
                       edge.code=0;  // we see the code later
                       _edge.push_back(edge); // it is a real new edge so save it
                       face.edge[k]=&_edge[_edge.size()-1]; // and keep the pointer
                       }
                   }
              // at that stage, the face has all its edge
              // check if the face still exists
              exist=false;
              face.nb_edge=4;  
              for(size_t m=0;m<_face.size();m++)// look if the face still exist
              if(isEqual(&_face[m],&face))  
                            {
                             cell.face[j]=&_face[m];exist=true;                    
                            } // if exist we keep the pointer
              if(!exist) // if not exist create it
                   {  
                   face.label=_face.size()+1;             
                   face.code=0;  // we see the code later
                   _face.push_back(face); // it is a real new face so save it
                   cell.face[j]=&_face[_face.size()-1]; // and keep the pointer
                    }  
               }    
         // at that stage we have all the face and edge now chesk the code
         cell.nb_face=6;   
         exist=false;
         for(size_t j=0;j<_cell.size();j++)// look if the cell still exist
             {    
             if(isEqual(&_cell[j],&cell))                 
                 {
                  exist=true;
                  if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {_cell[j].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}                  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {_cell[j].setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
                  if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {_cell[j].setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       _cell[j].code=ptr_el->code_physical;
                  }
             }      
         if(!exist) 
             {
              //cout<<"the cell (gmsh label)"<<ptr_el->label<< " does not exist"<<endl;  fflush(NULL);
              //cout<<"the edge are "<<cell.edge[0]->label<<" "<<cell.edge[1]->label<<" "<<cell.edge[2]->label<<" "<<endl;
              
              ptr_el->label=cell.label=_cell.size()+1;       
              if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {cell.setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}              
              if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {cell.setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
              if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {cell.setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}   
              if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       cell.code=ptr_el->code_physical;
              _cell.push_back(cell); // it is a real new cell so save it

             }  
         }    
    // treat  prism 
     if  (ptr_el->type_element==6) 
         {
          for(size_t j=0;j<3;j++)  //loop on the face for the quadrangle
             {
              FVFace3D face;     // construct a Face of triangle
              for(size_t k=0;k<4;k++)  // loop on the edge
                  {
                  FVEdge3D edge;  //seek the Edge               
                  size_t n1,n2,nd[4];
                  if(j==0) {nd[0]=0;nd[1]=1;nd[2]=4;nd[3]=3;}
                  if(j==1) {nd[0]=0;nd[1]=2;nd[2]=5;nd[3]=3;}
                  if(j==2) {nd[0]=1;nd[1]=2;nd[2]=5;nd[3]=4;}                
                  n1=nd[k];n2=nd[(k+1)%4];               
                  edge.firstVertex=&(_vertex[ptr_el->node[n1]-1]);
                  edge.secondVertex=&(_vertex[ptr_el->node[n2]-1]);
                  edge.nb_vertex=2;
                  exist=false;
                  for(size_t m=0;m<_edge.size();m++)// look if the edge still exist
                         if(isEqual(&_edge[m],&edge))  
                            {
                             face.edge[k]=&_edge[m];exist=true;                    
                            } // if exist we keep the pointer
                  if(!exist) // if not exist create it
                      {  
                       edge.label=_edge.size()+1;             
                       edge.code=0;  // we see the code later
                       _edge.push_back(edge); // it is a real new edge so save it
                       face.edge[k]=&_edge[_edge.size()-1]; // and keep the pointer
                       }
                   }
              // at that stage, the face has all its edge
              // check if the face still exists
              exist=false;
              face.nb_edge=4;  
              for(size_t m=0;m<_face.size();m++)// look if the face still exist
              if(isEqual(&_face[m],&face))  
                            {
                             cell.face[j]=&_face[m];exist=true;                    
                            } // if exist we keep the pointer
              if(!exist) // if not exist create it
                   {  
                   face.label=_face.size()+1;             
                   face.code=0;  // we see the code later
                   _face.push_back(face); // it is a real new face so save it
                   cell.face[j]=&_face[_face.size()-1]; // and keep the pointer
                    }  
               }  
          for(size_t j=3;j<5;j++)  //loop on the face for the triangle
             {
              FVFace3D face;     // construct a Face of triangle
              for(size_t k=0;k<3;k++)  // loop on the edge
                  {
                  FVEdge3D edge;  //seek the Edge
                  size_t n1,n2,nd[3];
                  if(j==3) {nd[0]=3;nd[1]=4;nd[2]=5;}
                  if(j==4) {nd[0]=0;nd[1]=1;nd[2]=2;}            
                  n1=nd[k];n2=nd[(k+1)%3];                                      
                  edge.firstVertex=&(_vertex[ptr_el->node[n1]-1]);
                  edge.secondVertex=&(_vertex[ptr_el->node[n2]-1]);
                  edge.nb_vertex=2;
                  exist=false;
                  for(size_t m=0;m<_edge.size();m++)// look if the edge still exist
                         if(isEqual(&_edge[m],&edge))  
                            {
                             face.edge[k]=&_edge[m];exist=true;                    
                            } // if exist we keep the pointer
                  if(!exist) // if not exist create it
                      {  
                       edge.label=_edge.size()+1;             
                       edge.code=0;  // we see the code later
                       _edge.push_back(edge); // it is a real new edge so save it
                       face.edge[k]=&_edge[_edge.size()-1]; // and keep the pointer
                       }
                   }
              // at that stage, the face has all its edge
              // check if the face still exists
              exist=false;
              face.nb_edge=3;  
              for(size_t m=0;m<_face.size();m++)// look if the face still exist
              if(isEqual(&_face[m],&face))  
                            {
                             cell.face[j]=&_face[m];exist=true;                    
                            } // if exist we keep the pointer
              if(!exist) // if not exist create it
                   {  
                   face.label=_face.size()+1;             
                   face.code=0;  // we see the code later
                   _face.push_back(face); // it is a real new face so save it
                   cell.face[j]=&_face[_face.size()-1]; // and keep the pointer
                    }  
               }               
               
         // at that stage we have all the face and edge now chesk the code
         cell.nb_face=5;   
         exist=false;
         for(size_t j=0;j<_cell.size();j++)// look if the cell still exist
             {    
             if(isEqual(&_cell[j],&cell))                 
                 {
                  exist=true;
                  if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {_cell[j].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}                  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {_cell[j].setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
                  if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {_cell[j].setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       _cell[j].code=ptr_el->code_physical;
                  }
             }      
         if(!exist) 
             {
              //cout<<"the cell (gmsh label)"<<ptr_el->label<< " does not exist"<<endl;  fflush(NULL);
              //cout<<"the edge are "<<cell.edge[0]->label<<" "<<cell.edge[1]->label<<" "<<cell.edge[2]->label<<" "<<endl;
              
              ptr_el->label=cell.label=_cell.size()+1;       
              if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {cell.setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}              
              if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {cell.setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
              if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {cell.setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}   
              if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       cell.code=ptr_el->code_physical;
              _cell.push_back(cell); // it is a real new cell so save it

             }  
         }   
    // treat  pyramid 
    if  (ptr_el->type_element==7) 
         {
           for(size_t j=0;j<1;j++)  //loop on the face for the quadrangle
             {
              FVFace3D face;     // construct a Face of triangle
              for(size_t k=0;k<4;k++)  // loop on the edge
                  {
                  FVEdge3D edge;  //seek the Edge               
                  size_t n1,n2,nd[4];
                  if(j==0) {nd[0]=0;nd[1]=1;nd[2]=2;nd[3]=3;}             
                  n1=nd[k];n2=nd[(k+1)%4];               
                  edge.firstVertex=&(_vertex[ptr_el->node[n1]-1]);
                  edge.secondVertex=&(_vertex[ptr_el->node[n2]-1]);
                  edge.nb_vertex=2;
                  exist=false;
                  for(size_t m=0;m<_edge.size();m++)// look if the edge still exist
                         if(isEqual(&_edge[m],&edge))  
                            {
                             face.edge[k]=&_edge[m];exist=true;                    
                            } // if exist we keep the pointer
                  if(!exist) // if not exist create it
                      {  
                       edge.label=_edge.size()+1;             
                       edge.code=0;  // we see the code later
                       _edge.push_back(edge); // it is a real new edge so save it
                       face.edge[k]=&_edge[_edge.size()-1]; // and keep the pointer
                       }
                   }
              // at that stage, the face has all its edge
              // check if the face still exists
              exist=false;
              face.nb_edge=4;  
              for(size_t m=0;m<_face.size();m++)// look if the face still exist
              if(isEqual(&_face[m],&face))  
                            {
                             cell.face[j]=&_face[m];exist=true;                    
                            } // if exist we keep the pointer
              if(!exist) // if not exist create it
                   {  
                   face.label=_face.size()+1;             
                   face.code=0;  // we see the code later
                   _face.push_back(face); // it is a real new face so save it
                   cell.face[j]=&_face[_face.size()-1]; // and keep the pointer
                    }  
               }  
          for(size_t j=1;j<5;j++)  //loop on the face for the triangle
             {
              FVFace3D face;     // construct a Face of triangle
              for(size_t k=0;k<3;k++)  // loop on the edge
                  {
                  FVEdge3D edge;  //seek the Edge
                  size_t n1,n2,nd[3];
                  if(j==3) {nd[0]=4;nd[1]=0;nd[2]=1;}
                  if(j==4) {nd[0]=4;nd[1]=1;nd[2]=2;}    
                  if(j==3) {nd[0]=4;nd[1]=2;nd[2]=3;}
                  if(j==4) {nd[0]=4;nd[1]=3;nd[2]=0;}                    
                  n1=nd[k];n2=nd[(k+1)%3];                                      
                  edge.firstVertex=&(_vertex[ptr_el->node[n1]-1]);
                  edge.secondVertex=&(_vertex[ptr_el->node[n2]-1]);
                  edge.nb_vertex=2;
                  exist=false;
                  for(size_t m=0;m<_edge.size();m++)// look if the edge still exist
                         if(isEqual(&_edge[m],&edge))  
                            {
                             face.edge[k]=&_edge[m];exist=true;                    
                            } // if exist we keep the pointer
                  if(!exist) // if not exist create it
                      {  
                       edge.label=_edge.size()+1;             
                       edge.code=0;  // we see the code later
                       _edge.push_back(edge); // it is a real new edge so save it
                       face.edge[k]=&_edge[_edge.size()-1]; // and keep the pointer
                       }
                   }
              // at that stage, the face has all its edge
              // check if the face still exists
              exist=false;
              face.nb_edge=3;  
              for(size_t m=0;m<_face.size();m++)// look if the face still exist
              if(isEqual(&_face[m],&face))  
                            {
                             cell.face[j]=&_face[m];exist=true;                    
                            } // if exist we keep the pointer
              if(!exist) // if not exist create it
                   {  
                   face.label=_face.size()+1;             
                   face.code=0;  // we see the code later
                   _face.push_back(face); // it is a real new face so save it
                   cell.face[j]=&_face[_face.size()-1]; // and keep the pointer
                    }  
               }               
         // at that stage we have all the face and edge now chesk the code
         cell.nb_face=5;   
         exist=false;
         for(size_t j=0;j<_cell.size();j++)// look if the cell still exist
             {    
             if(isEqual(&_cell[j],&cell))                 
                 {
                  exist=true;
                  if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {_cell[j].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}                  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {_cell[j].setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
                  if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {_cell[j].setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}  
                  if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       _cell[j].code=ptr_el->code_physical;
                  }
             }      
         if(!exist) 
             {
              //cout<<"the cell (gmsh label)"<<ptr_el->label<< " does not exist"<<endl;  fflush(NULL);
              //cout<<"the edge are "<<cell.edge[0]->label<<" "<<cell.edge[1]->label<<" "<<cell.edge[2]->label<<" "<<endl;
              
              ptr_el->label=cell.label=_cell.size()+1;       
              if (ptr_el->code_physical/MINUS_ONE_DIM==3)               
                      {cell.setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}              
              if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
                      {cell.setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}
              if (ptr_el->code_physical/MINUS_ONE_DIM==1)
                      {cell.setCode2Face(ptr_el->code_physical%MINUS_ONE_DIM);}   
              if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
                       cell.code=ptr_el->code_physical;
              _cell.push_back(cell); // it is a real new cell so save it

             }  
         }    
     }   
     
_nb_edge=_edge.size();
_nb_face=_face.size();
_nb_cell=_cell.size();  
// cout<<"step2"<<endl;  
// all the edge and cell are created    
// now we treat the code cascade backward

for(size_t cont=pos_dim3+1;cont>0;cont--) // we skip of 1 becaus NEGATIVE NUMBER DOES NOT ESXIST with size_t
   {  
    size_t i=cont-1; // astuce to allow deacresing counter   
    GMElement *ptr_el=m.getElement(i); 
// We treat the 2D element 
    if  (ptr_el->type_element==2 ||ptr_el->type_element==3 )  // treat   a triangle or a quadrangle
        {
                    // we set the vertex code 
        if (ptr_el->code_physical/MINUS_ONE_DIM==2)
              {_face[ptr_el->label-1].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}  
              // we set the edge code             
        if (ptr_el->code_physical/MINUS_ONE_DIM==1)
              {_face[ptr_el->label-1].setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}  
              // we set the facex code 
        if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
              _face[ptr_el->label-1].code=ptr_el->code_physical;   
        }
// we treat the edge
    if(ptr_el->type_element==1)
        { 
        if (ptr_el->code_physical/MINUS_ONE_DIM==1)
              {_edge[ptr_el->label-1].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}  // we set the Vertex code 
        if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
              _edge[ptr_el->label-1].code=ptr_el->code_physical;   // we set the edge code 
        }       
//we treat the node       
    if(ptr_el->type_element==15) 
           {
            _vertex[ptr_el->label-1].code=ptr_el->code_physical;  // we set the Vertex code 
           }
     } 
FVMesh3D::complete_data();
}








