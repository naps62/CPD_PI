#include "Gmsh.h"
#include <cstring>
#include <cstdlib>


Gmsh::Gmsh()
{
  _if_is_open=false,_of_is_open=false; _dim=0; _nb_node=0;_nb_element=0;
}  

Gmsh::Gmsh(const char *filename)
{
    Gmsh::readMesh(filename);
    
}

Gmsh::~Gmsh()
{
  Gmsh::close();
} 
/*--------- read and write mesh -----------*/
void Gmsh::readMesh(const char *filename)
{
if(_if_is_open)
            {
             cout<<" error opening file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" an open file exists yet"<<endl;
             exit(0);
            }    
_if.open(filename);
if(!_if.is_open())
            {
             cout<<" error opening file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" file does not exist"<<endl;
             exit(0);
            }
_if_is_open=true;    
_dim=0;
size_t filesize;
_if.seekg( -1, ios_base::end );
filesize=_if.tellg();
char *ptr,*file_string;
file_string=(char *)malloc(filesize+2);
size_t nb_code,pos;
_if.seekg(0);
pos=0;
while (!_if.eof())// read the file
      {
      file_string[pos]=(char) _if.get();
      pos++;
      }     
      
// the file is now in the string
ptr=file_string;
ptr= strstr( ptr,"$Nodes")+6;
_nb_node= strtod(ptr, &ptr);
_node.resize(_nb_node); 
for(size_t i=0;i<_nb_node;i++)
    {
     _node[i].label  = strtod(ptr, &ptr);
     _node[i].coord.x= strtod(ptr, &ptr);
     _node[i].coord.y= strtod(ptr, &ptr);     
     _node[i].coord.z= strtod(ptr, &ptr);     
    }
ptr= strstr( ptr,"$Elements")+9;
_nb_element= strtod(ptr, &ptr);
_element.resize(_nb_element); 
for(size_t i=0;i<_nb_element;i++)
    {
     _element[i].label  = strtod(ptr, &ptr);
     _element[i].type_element= strtod(ptr, &ptr);
     nb_code = strtod(ptr, &ptr); 
     if(nb_code < 2)
         {cout<<"invalid number of code in msh file"<<endl;exit(0);}
     _element[i].code_physical= strtod(ptr, &ptr);     
     _element[i].code_elementary= strtod(ptr, &ptr);     
     for(size_t j=0;j<nb_code-2;j++)  pos=strtod(ptr, &ptr);
     switch(_element[i].type_element)
         {
         case 15:   // node
         _element[i].nb_node=1;
         _element[i].dim=0;
         break;
         case 1:  // line
         case 8:
         _element[i].nb_node=2;
         _element[i].dim=1;
         break;
         case 2:  //triangle
         case 9:    
         _element[i].nb_node=3;
         _element[i].dim=2;
         break;
         case 3:  // quadrangle
         case 10:    
         _element[i].nb_node=4;
         _element[i].dim=2;
         break;
         case 4:  //tetrahedron
         case 11:
         _element[i].nb_node=4;
         _element[i].dim=3;
         break;
         case 5:  // hexahedron
         case 12:
         _element[i].nb_node=8;
         _element[i].dim=3;
         break;
         case 6:  // prism
         case 13:
         _element[i].nb_node=6;
         _element[i].dim=3;
         break;
         case 7:  // pyramid
         case 14:
         _element[i].nb_node=5;
         _element[i].dim=3;
         break;    
         default:
         cout<<"error code element in file msh"<<endl; exit(0);
         break;
         }
    if (_dim<  _element[i].dim) _dim= _element[i].dim;  
    for(size_t j=0;j<_element[i].nb_node;j++) _element[i].node[j]= strtod(ptr, &ptr);
    }
free(file_string);    
// we have fill all the gmsh structure        
}

void Gmsh::writeMesh(const char *filename)
{
if(_of_is_open==true)
    { 
    cout<<"a file is still open: close it before writing a mesh"<<endl;
    }

if((_nb_node==0) || (_nb_element==0))
            {
             cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" there is no mesh to save"<<endl;
             return;
            }       
_of.open(filename);
if(!_of.is_open())
            {
             cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
             cout<<" can not create the file"<<endl;
             return;
            }
_of_is_open=true;_nb_save=0;           
_of<<"$MeshFormat"<<endl;
_of<<"2.2 0 8"<<endl;
_of<<"$EndMeshFormat"<<endl;
_of<<"$Nodes"<<endl;
_of<<_nb_node<<endl;
for(size_t i=0;i<_nb_node;i++)
    {
    _of<<_node[i].label<< setw(FVCHAMP)<<_node[i].coord.x<<setw(FVCHAMP);
    _of<<setw(FVCHAMP)<<_node[i].coord.y<<setw(FVCHAMP)<<_node[i].coord.z<<endl;
    }
_of<<"$EndNodes"<<endl;
_of<<"$Elements"<<endl;
_of<<_nb_element<<endl;
for(size_t i=0;i<_nb_element;i++)
    {
    _of<<_element[i].label<< setw(FVCHAMP)<<_element[i].type_element<< setw(FVCHAMP)<<2;
    _of<<setw(FVCHAMP)<<_element[i].code_physical<< setw(FVCHAMP)<<_element[i].code_elementary;
    for(size_t j=0;j<_element[i].nb_node;j++)
            _of<<setw(FVCHAMP)<<_element[i].node[j];
    _of<<endl;
      }
_of<<"$EndElements"<<endl;
}

void Gmsh::close()
{
 if(_if_is_open) {_if.close();_if_is_open=false;}
 if(_of_is_open) {_of.close();_of_is_open=false;} 
} 



void Gmsh::writeVector(FVVect<fv_float> &u, const size_t type, const char *name, fv_float time)
{
if(!_of_is_open)
    { 
    cout<<"write the mesh file first"<<endl;
    }   
if(type== VERTEX)
    {
    _of<<"$NodeData"<<endl;   
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl; 
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"1"<<endl;
    _of<<_nb_node<<endl;
    for(size_t i=0;i<_nb_node;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i]<<endl;
    _of<<"$EndNodeData"<<endl;   
    }
     
if(type== CELL)    
    {
    _of<<"$ElementData"<<endl; 
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl;
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"1"<<endl;
    _of<<_nb_element<<endl;
    for(size_t i=0;i<_nb_element;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i]<<endl;
    _of<<"$EndElementData"<<endl;   
    }
 
}
    
void Gmsh::writeVector(const FVVect<fv_float> &u,const FVVect<fv_float> &v,  const size_t type,const char *name, fv_float time) 
{
if(!_of_is_open)
    { 
    cout<<"write the mesh file first"<<endl;
    } 
if(type== VERTEX)
    {
    _of<<"$NodeData"<<endl;   
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl; 
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_node<<endl;
    for(size_t i=0;i<_nb_node;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i]<<setw(FVCHAMP)<<v[i]<<setw(FVCHAMP)<<"0"<<endl;
    _of<<"$EndNodeData"<<endl;   
    }
    
if(type== CELL)    
    {
    _of<<"$ElementData"<<endl; 
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl;
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_element<<endl;
    for(size_t i=0;i<_nb_element;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i]<<setw(FVCHAMP)<<v[i]<<setw(FVCHAMP)<<"0"<<endl;
    _of<<"$EndElementData"<<endl;   
    }    
}
       
void Gmsh::writeVector(const FVVect<fv_float> &u,const FVVect<fv_float> &v, const FVVect<fv_float> &w, const size_t type,const char *name, fv_float time)   
{
if(!_of_is_open)
    { 
    cout<<"write the mesh file first"<<endl; 
    } 
if(type== VERTEX)
    {
    _of<<"$NodeData"<<endl;   
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl; 
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_node<<endl;
    for(size_t i=0;i<_nb_node;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i]<<setw(FVCHAMP)<<v[i]<<setw(FVCHAMP)<<w[i]<<endl;
    _of<<"$EndNodeData"<<endl;   
    }
    
if(type== CELL)    
    {
    _of<<"$ElementData"<<endl; 
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl;
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_element<<endl;
    for(size_t i=0;i<_nb_element;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i]<<setw(FVCHAMP)<<v[i]<<setw(FVCHAMP)<<w[i]<<endl;
    _of<<"$EndElementData"<<endl;   
    }    
}


void Gmsh::writeVector(FVVect<FVPoint1D<fv_float> >&u, const size_t type, const char *name, fv_float time)
{
if(!_of_is_open)
    { 
    cout<<"write the mesh file first"<<endl;
    }   
if(type== VERTEX)
    {   
    _of<<"$NodeData"<<endl;   
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl; 
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"1"<<endl;
    _of<<_nb_node<<endl;
    for(size_t i=0;i<_nb_node;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i].x<<endl;
    _of<<"$EndNodeData"<<endl;   
    }
    
if(type== CELL)    
    {    
    _of<<"$ElementData"<<endl; 
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl;
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"1"<<endl;
    _of<<_nb_element<<endl;
    for(size_t i=0;i<_nb_element;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i].x<<endl;
    _of<<"$EndElementData"<<endl;   
    }
 
}
    
void Gmsh::writeVector(FVVect<FVPoint2D<fv_float> >&u, const size_t type,const char *name, fv_float time) 
{
if(!_of_is_open)
    { 
    cout<<"write the mesh file first"<<endl;
    } 
if(type== VERTEX)
    {
    _of<<"$NodeData"<<endl;   
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl; 
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_node<<endl;
    for(size_t i=0;i<_nb_node;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i].x<<setw(FVCHAMP)<<u[i].y<<setw(FVCHAMP)<<"0"<<endl;
    _of<<"$EndNodeData"<<endl;   
    }
    
if(type== CELL)    
    {  
    _of<<"$ElementData"<<endl; 
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl;
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_element<<endl;
    for(size_t i=0;i<_nb_element;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i].x<<setw(FVCHAMP)<<u[i].y<<setw(FVCHAMP)<<"0"<<endl;
    _of<<"$EndElementData"<<endl;   
    }    
}
       
void Gmsh::writeVector(FVVect<FVPoint3D<fv_float> >&u, const size_t type,const char *name, fv_float time)   
{
if(!_of_is_open)
    { 
    cout<<"write the mesh file first"<<endl; 
    } 
if(type== VERTEX)
    {
    _of<<"$NodeData"<<endl;   
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl; 
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_node<<endl;
    for(size_t i=0;i<_nb_node;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i].x<<setw(FVCHAMP)<<u[i].y<<setw(FVCHAMP)<<u[i].z<<endl;
    _of<<"$EndNodeData"<<endl;   
    }
    
if(type== CELL)    
    {
    _of<<"$ElementData"<<endl; 
    _of<<"1"<<endl;
    _of<<"\""<<name<<"\""<<endl;
    _of<<"1"<<endl;
    _of<<time<<endl;
    _of<<"3"<<endl;
    _of<<_nb_save++<<endl;
    _of<<"3"<<endl;
    _of<<_nb_element<<endl;
    for(size_t i=0;i<_nb_element;i++)
        _of<<setw(FVCHAMPINT)<<i+1<<setw(FVCHAMP)<<u[i].x<<setw(FVCHAMP)<<u[i].y<<setw(FVCHAMP)<<u[i].z<<endl;
    _of<<"$EndElementData"<<endl;   
    }    
}






















































//
//------------- CONVERTISOR  -----------------
//
void Gmsh::FVMesh2Gmsh(FVMesh1D &m) // constructor which convert a FVMesh1D into a gmah struct
{
_dim=1;   // a one dimension mesh
FVVertex1D* ptr_v;    
_nb_node=m.getNbVertex();
_node.resize(_nb_node); 
for((ptr_v=m.beginVertex());(ptr_v=m.nextVertex());)
    {
     size_t i=ptr_v->label;
     _node[i-1].label  = i; 
     _node[i-1].coord.x= ptr_v->coord.x;
     _node[i-1].coord.y= 0;     
     _node[i-1].coord.z= 0; 
    }
FVCell1D *ptr_c;
_nb_element=m.getNbCell();
_element.resize(_nb_element);
for((ptr_c=m.beginCell());(ptr_c=m.nextCell());)
    {
    size_t i=ptr_c->label;
    _element[i-1].label=i;
    _element[i-1].type_element=1; // we use line2
    _element[i-1].code_physical=ptr_c->code;
    _element[i-1].code_elementary=ptr_c->code;
    _element[i-1].dim=_dim;  
    _element[i-1].nb_node=2;
    _element[i-1].node[0]= ptr_c->firstVertex->label;
    _element[i-1].node[1]= ptr_c->secondVertex->label;
    }
    
}
     
     
   
void Gmsh::FVMesh2Gmsh(FVMesh2D &m) // constructor which convert a FVMesh2D into a gmah struct 
{
_dim=2;   // a two dimension mesh
FVVertex2D* ptr_v;    
_nb_node=m.getNbVertex();
_node.resize(_nb_node); 
for((ptr_v=m.beginVertex());(ptr_v=m.nextVertex());)
    {
     size_t i=ptr_v->label;
     _node[i-1].label  = i; 
     _node[i-1].coord.x= ptr_v->coord.x;
     _node[i-1].coord.y= ptr_v->coord.y;     
     _node[i-1].coord.z= 0; 
    }
FVCell2D *ptr_c;
fv_float order[4],doux;
size_t noux;
FVPoint2D<fv_float> Paux;
_nb_element=m.getNbCell();
_element.resize(_nb_element);
for((ptr_c=m.beginCell());(ptr_c=m.nextCell());)
    {
    size_t i=ptr_c->label;
    _element[i-1].label=i;
    switch(ptr_c->nb_vertex)
        {
        case 3:
        _element[i-1].type_element=2; // we use triangle
        break;
        case 4:
        _element[i-1].type_element=3; // we use quadrangle
        break;
        default:
        cout<<"cell does not correspond to a gmsh element. Abort convertion"<<endl;
        _nb_node=0;_nb_element=0;
        return;
        }    
    _element[i-1].code_physical=ptr_c->code;
    _element[i-1].code_elementary=ptr_c->code;
    _element[i-1].dim=_dim;  
    _element[i-1].nb_node=ptr_c->nb_vertex;
    for(size_t j=0;j<ptr_c->nb_vertex;j++)
        {
        _element[i-1].node[j]=ptr_c->vertex[j]->label; 
        Paux.x=_node[_element[i-1].node[j]-1].coord.x-ptr_c->centroid.x;
        Paux.y=_node[_element[i-1].node[j]-1].coord.y-ptr_c->centroid.y;
        Paux/=Norm(Paux);
        order[j]=(1-Paux.x)*0.5;if(Paux.y<0) order[j]*=-1;   
        }  
    // reordering the node (bubble sort) to cast in the gmah framework  following order
    for(size_t j=ptr_c->nb_vertex-1;j>0;j--)
        {
        for(size_t k=0;k<j;k++)
            {
            if(order[k+1]<order[k])
                 {
                  doux=order[k+1];noux=_element[i-1].node[k+1];
                  order[k+1]=order[k];_element[i-1].node[k+1]=_element[i-1].node[k];
                  order[k]=doux;_element[i-1].node[k]=noux;
                 }
            }
        }   
    } 
 
}
     
void Gmsh::FVMesh2Gmsh(FVMesh3D &m) // constructor which convert a FVMesh3D into a gmah struct
{
_dim=3;   // a three dimension mesh
FVVertex3D* ptr_v;
FVEdge3D* ptr_e;
FVFace3D* ptr_f;
FVCell3D *ptr_c;
FVPoint3D<fv_float> Paux,t1,t2;
fv_float order[5],doux;
size_t noux;
size_t it;
_nb_node=m.getNbVertex();
_node.resize(_nb_node); 
//cout<<"find "<<_nb_node<<" nodes"<<endl;fflush(NULL);
for((ptr_v=m.beginVertex());(ptr_v=m.nextVertex());)
    {
     size_t i=ptr_v->label;
     _node[i-1].label  = i; 
     _node[i-1].coord.x= ptr_v->coord.x;
     _node[i-1].coord.y= ptr_v->coord.y;     
     _node[i-1].coord.z= ptr_v->coord.z;        
    }
_nb_element=m.getNbCell();
_element.resize(_nb_element);
//cout<<"find "<<_nb_element<<" elements"<<endl;fflush(NULL);
for((ptr_c=m.beginCell());(ptr_c=m.nextCell());)
    {
    size_t i=ptr_c->label;
    _element[i-1].label=i;
    _element[i-1].code_physical=ptr_c->code;
    _element[i-1].code_elementary=ptr_c->code;
    _element[i-1].dim=_dim;  
    _element[i-1].nb_node=ptr_c->nb_vertex;
    //cout<<"element "<<i<<" with "<<ptr_c->nb_vertex<<" nodes"<<endl;fflush(NULL);
    switch(_element[i-1].nb_node)
        {
        case 4:   
        _element[i-1].type_element=4; // we use tetrahedron
        for(size_t j=0;j<4;j++)  // the order is not important
             _element[i-1].node[j]=ptr_c->vertex[j]->label;
        break;
        case 5:  
        _element[i-1].type_element=7; // pyramid
        it=0;
        for((ptr_f=ptr_c->beginFace());(ptr_f=ptr_c->nextFace());it++)
            {  
            if( ptr_c->face[it]->nb_vertex==4) break; 
            }
        if (it>=ptr_c->nb_face)  {cout<<" not a pyramidal element"<<endl;return;}
        ptr_f=ptr_c->face[it];      
        // ok, I have the 4-vertex-face now I load the 5 nodes
        for(size_t j=0;j<5;j++)  _element[i-1].node[j]=ptr_c->vertex[j]->label;
        // the "positive" vertex is the top of the pyramid
        for(size_t j=0;j<5;j++)
            {
            Paux=_node[_element[i-1].node[j]-1].coord-ptr_c->centroid;
            doux=Paux*ptr_c->cell2face[it];
            if(doux<0) {noux=_element[i-1].node[j];_element[i-1].node[j]=_element[i-1].node[4];_element[i-1].node[4]=noux;}
            } 
        // at that stge the     node[4] is the top vertex
        // we have to sort the four first points.
        t1.x=ptr_c->cell2face[it].y;t1.y=-ptr_c->cell2face[it].y;t1.z=0;t1/=Norm(t1);// first tangente
        t2.x=ptr_c->cell2face[it].z;t2.z=-ptr_c->cell2face[it].x;t2.y=0;t2/=Norm(t2);// second tangente   
        for(size_t j=0;j<3;j++)
           {
            Paux=_node[_element[i-1].node[j]-1].coord-ptr_f->centroid;
            Paux/=Norm(Paux);
            order[j]=(1-Paux*t1)*0.5;if(Paux*t2<0) order[j]*=-1;   
            }  
        for(size_t j=3;j>0;j--)
             {
             for(size_t k=0;k<j;k++)
                {
                if(order[k+1]<order[k])
                     {
                      doux=order[k+1];noux=_element[i-1].node[k+1];
                      order[k+1]=order[k];_element[i-1].node[k+1]=_element[i-1].node[k];
                      order[k]=doux;_element[i-1].node[k]=noux;
                     }
                }
             }        
        break;
        case 6:
        _element[i-1].type_element=6; // prism
        it=0;
        for((ptr_f=ptr_c->beginFace());(ptr_f=ptr_c->nextFace());it++)
               {if(ptr_f->nb_vertex==3) break;}
        for(size_t j=0;j<3;j++)  
             {_element[i-1].node[j]=ptr_f->vertex[j]->label;_element[i-1].node[j+3]=0;}          
        // at that stage the basement face is ordered
        // we see the complementary point
        for(size_t j=0;j<3;j++)
            {
            it=_element[i-1].node[j]; // take a node of the reference base
            for((ptr_f=ptr_c->beginFace());(ptr_f=ptr_c->nextFace());)
                for((ptr_e=ptr_f->beginEdge());(ptr_e=ptr_f->nextEdge());)
                {
                 if((ptr_e->firstVertex->label!=it) && (ptr_e->secondVertex->label!=it)) continue;
                 if(ptr_e->firstVertex->label==it) noux=ptr_e->secondVertex->label; else noux=ptr_e->firstVertex->label;
                  bool still_exist=false;
                 for(size_t k=0;k<3;k++)
                     if(noux==_element[i-1].node[k]) still_exist=true;
                 if(!still_exist) {_element[i-1].node[j+3]=noux;}    
                }
            }
        // check if we have really 6 nodes             
        for(size_t j=0;j<6;j++) 
            {
            if(! _element[i-1].node[j])
               {
                cout<<" Not a prism element: abort convertion"<<endl; return;
                }
            }    
        break;    
        case 8:
        _element[i-1].type_element=5; // cube
        ptr_f=ptr_c->face[0]; // reference face
        for(size_t j=0;j<4;j++)  _element[i-1].node[j]=ptr_f->vertex[j]->label;
       // we have to sort the four first points.
        t1= _node[_element[i-1].node[0]-1].coord-ptr_f->centroid;t1/=Norm(t1);// first tangente
        t2=CrossProduct(t1,ptr_f->normal[0]);t2/=Norm(t2);// second tangente   
        for(size_t j=0;j<4;j++)
           {
            Paux=_node[_element[i-1].node[j]-1].coord-ptr_f->centroid;
            Paux/=Norm(Paux);
            order[j]=(1-Paux*t1)*0.5;if(Paux*t2<0) order[j]*=-1;  
            }  
            
        for(size_t j=3;j>0;j--)
             {
             for(size_t k=0;k<j;k++)
                {
                if(order[k+1]<order[k])
                     {
                      doux=order[k+1];noux=_element[i-1].node[k+1];
                      order[k+1]=order[k];_element[i-1].node[k+1]=_element[i-1].node[k];
                      order[k]=doux;_element[i-1].node[k]=noux;
                     }
                }
             }           
        // at that stage the basement face is ordered
        // we see the complementary point
        for(size_t j=0;j<4;j++)
            {
            it=_element[i-1].node[j]; // take a node of the reference base
            for((ptr_f=ptr_c->beginFace());(ptr_f=ptr_c->nextFace());)
                for((ptr_e=ptr_f->beginEdge());(ptr_e=ptr_f->nextEdge());)
                {
                 if((ptr_e->firstVertex->label!=it) && (ptr_e->secondVertex->label!=it)) continue;
                 if(ptr_e->firstVertex->label==it) noux=ptr_e->secondVertex->label; else noux=ptr_e->firstVertex->label;
                 bool still_exist=false;
                 for(size_t k=0;k<4;k++)
                     if(noux==_element[i-1].node[k]) still_exist=true;
                 if(!still_exist) _element[i-1].node[j+4]=noux;    
                }
            }
        break;           
        default:
        cout<<"cell does not correspond to a gmsh element. Abort convertion"<<endl;
        _nb_node=0;_nb_element=0;
        return;
        } 
    }      
}






    
