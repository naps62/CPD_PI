#ifndef __FVCELL3D_H
#define __FVCELL3D_H
#include <vector>
#include "FVPoint3D.h"
#include "FVLib_config.h"
using namespace std;

class FVVertex3D;
class FVEdge3D;
class FVFace3D;
class FVCell3D
{
public:
FVPoint3D<fv_float> centroid;
size_t label,code,nb_vertex,nb_face;
size_t pos_f,pos_v;
fv_float surface,volume;
FVVertex3D* vertex[NB_VERTEX_PER_CELL_3D] ; // the  vertices
FVFace3D* face[NB_FACE_PER_CELL_3D];     // the face
FVPoint3D<fv_float>  cell2face[NB_FACE_PER_CELL_3D]; // normal exterior for each face


     FVCell3D(){nb_vertex=0;nb_face=0;label=0;}
    ~FVCell3D(){;}  
    
     FVVertex3D* beginVertex(){pos_v=0;if(pos_v<nb_vertex) return(vertex[0]);else return(NULL);}
     FVVertex3D* nextVertex(){if(pos_v<nb_vertex) return(vertex[pos_v++]);else return(NULL);}  
     FVFace3D* beginFace(){pos_f=0;if(pos_f<nb_face) return(face[0]);else return(NULL);};
     FVFace3D* nextFace(){if(pos_f<nb_face) return(face[pos_f++]);else return(NULL);}     
     FVPoint3D<fv_float> getCell2Face(){return cell2face[pos_f-1];}
     void setCode2Face(size_t val=0)
         {for(size_t i=0;i<nb_face;i++) 
          if(face[i]) face[i]->code=val;}     
     void setCode2Edge(size_t val=0)
         {for(size_t i=0;i<nb_face;i++) 
          if(face[i]) face[i]->setCode2Edge(val);}
     void setCode2Vertex(size_t val=0)
         { 
             for(size_t i=0;i<nb_face;i++) 
                 if(face[i]) 
                 {face[i]->setCode2Vertex(val);}   
         } 

private:

};



inline bool isEqual(FVCell3D *c1, FVCell3D *c2)
     {
      bool is_equal,is_equal_total;
      if(c1->nb_face!=c2->nb_face) return false;
      is_equal_total=true;
      for(size_t i=0;i<c1->nb_face;i++)
         {
          is_equal=false;   
          for(size_t j=0;j<c2->nb_face;j++)
              if(c1->face[i]->label==c2->face[j]->label) is_equal=true;
          is_equal_total=is_equal_total && is_equal;    
         }
       return(is_equal_total);  
     }

#endif // define _FVCELL3D
