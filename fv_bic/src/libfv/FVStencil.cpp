#include "FVStencil.h"

FVStencil::FVStencil(const FVStencil &st) // copy class
               {
                _nb_geometry=st._nb_geometry;
                _pos=st._pos;
                _reference_geometry=st._reference_geometry;
                _reference_type=st._reference_type;
                
                _geometry=new vector<void*> ;_type=new vector<size_t>;
                _geometry->resize(_nb_geometry);
                _type->resize(_nb_geometry);
                for(size_t i=0;i<_nb_geometry;i++)
                      {(*_geometry)[i]=(* st._geometry)[i];(*_type)[i]=(* st._type)[i];}  
                }
void FVStencil::show()
{
cout<<"======= Stencil composition====="<<endl;    
if (_reference_geometry==NULL) _reference_type=NULL_ENTITY;
cout<<"== Reference element =="<<endl;    
switch(_reference_type)   
   {
    case NULL_ENTITY: 
    cout<<"No reference  element"<<endl;    
    break;
    case FVVERTEX1D:
    cout<<"FVVertex1D element of label "<< ((FVVertex1D *)_reference_geometry)->label<<endl;
    break;
    case FVCELL1D:
    cout<<"FVCell1D element of label "<< ((FVCell1D *)_reference_geometry)->label<<endl;
    break;    
    case FVVERTEX2D:
    cout<<"FVVertex2D element of label "<< ((FVVertex2D *)_reference_geometry)->label<<endl;
    break;
    case FVEDGE2D:
    cout<<"FVEdge2D element of label "<< ((FVEdge2D *)_reference_geometry)->label<<endl;
    break;      
    case FVCELL2D:
    cout<<"FVCell2D element of label "<< ((FVCell2D *)_reference_geometry)->label<<endl;
    break;  
    case FVVERTEX3D:
    cout<<"FVVertex3D element of label "<< ((FVVertex3D *)_reference_geometry)->label<<endl;
    break;
    case FVEDGE3D:
    cout<<"FVEdge3D element of label "<< ((FVEdge3D *)_reference_geometry)->label<<endl;
    break;      
    case FVFACE3D:
    cout<<"FVFace3D element of label "<< ((FVFace3D *)_reference_geometry)->label<<endl;
    break;      
    case FVCELL3D:
    cout<<"FVCell3D element of label "<< ((FVCell3D *)_reference_geometry)->label<<endl;
    break;  
    default:
    cout<<" Error of element geometry type in FVStencil::show"<<endl;
    }
cout<<"== Stencil contain "<<    _nb_geometry <<" geometrical entity=="<<endl;
if(_geometry==NULL) cout<<"problem with the _geometry, memory not allocated"<<endl;fflush(NULL);
for(size_t i=0;i<_nb_geometry;i++)
    {
    cout<<"Geometrical entity "<<i<<" ";    
    switch((*_type)[i])   
        {
        case NULL_ENTITY:
        cout<<"WARNING: NULL type reference element"<<endl;   
        break;
        case FVVERTEX1D:
        cout<<"FVVertex1D element of label "<< ((FVVertex1D *)(*_geometry)[i])->label<<endl;
        break;
        case FVCELL1D:
        cout<<"FVCell1D element of label "<< ((FVCell1D *)(*_geometry)[i])->label<<endl;
        break;    
        case FVVERTEX2D:
        cout<<"FVVertex2D element of label "<< ((FVVertex2D *)(*_geometry)[i])->label<<endl;
        break;
        case FVEDGE2D:
        cout<<"FVEdge2D element of label "<< ((FVEdge2D *)(*_geometry)[i])->label<<endl;
        break;      
        case FVCELL2D:
        cout<<"FVCell2D element of label "<< ((FVCell2D *)(*_geometry)[i])->label<<endl;
        break;  
        case FVVERTEX3D:
        cout<<"FVVertex3D element of label "<< ((FVVertex3D *)(*_geometry)[i])->label<<endl;
        break;
        case FVEDGE3D:
        cout<<"FVEdge3D element of label "<< ((FVEdge3D *)(*_geometry)[i])->label<<endl;
        break;      
        case FVFACE3D:
        cout<<"FVFace3D element of label "<< ((FVFace3D *)(*_geometry)[i])->label<<endl;
        break;      
        case FVCELL3D:
        cout<<"FVCell3D element of label "<< ((FVCell3D *)(*_geometry)[i])->label<<endl;
        break;  
        default:
        cout<<" Error of element geometry type in FVStencil::show"<<endl;
        }
    }
}

void FVStencil::setReferenceGeometry(FVVertex1D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVVERTEX1D;    
}
void FVStencil::addStencil(FVVertex1D *ptr )
{   
_geometry->push_back((void *) ptr);
_type->push_back(FVVERTEX1D);
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVVertex2D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVVERTEX2D;    
}
void FVStencil::addStencil(FVVertex2D *ptr ) 
{
_geometry->push_back((void *) ptr);
_type->push_back(FVVERTEX2D);
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVVertex3D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVVERTEX3D;    
}
void FVStencil::addStencil(FVVertex3D *ptr )
{
_geometry->push_back((void *) ptr);
_type->push_back(FVVERTEX3D);
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVCell1D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVCELL1D;    
}
void FVStencil::addStencil(FVCell1D *ptr )
{
_geometry->push_back((void *) ptr);
_type->push_back(FVCELL1D);
_nb_geometry++;
//cout<<"FVCell1D element of label "<< ((FVCell1D *)(*_geometry)[_nb_geometry-1])->label<<endl;
}
void FVStencil::setReferenceGeometry(FVCell2D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVCELL2D;    
}
void FVStencil::addStencil(FVCell2D *ptr )
{
_geometry->push_back((void *) ptr);
_type->push_back(FVCELL2D);
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVCell3D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVCELL3D;    
}
void FVStencil::addStencil(FVCell3D *ptr )
{
_geometry->push_back((void *) ptr);
_type->push_back(FVCELL3D);
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVEdge2D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVEDGE2D;    
} 
void FVStencil::addStencil(FVEdge2D *ptr )
{
_geometry->push_back((void *) ptr);
_type->push_back(FVEDGE2D);
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVEdge3D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVEDGE3D;    
}
void FVStencil::addStencil(FVEdge3D *ptr )  
{
_geometry->push_back((void *) ptr);  
_type->push_back(FVEDGE3D); 
_nb_geometry++;
}
void FVStencil::setReferenceGeometry(FVFace3D *ptr )
{
_reference_geometry=(void *) ptr;
_reference_type=FVFACE3D;    
}
void FVStencil::addStencil(FVFace3D *ptr )  
{
_geometry->push_back((void *) ptr);
_type->push_back(FVFACE3D);
_nb_geometry++;
} 

 



