#ifndef __FVEDGE3D_H
#define __FVEDGE3D_H
#include <vector>
#include "FVPoint3D.h"
#include "FVLib_config.h"
using namespace std;


class FVVertex3D;
class FVEdge3D
{
public:
public:
FVPoint3D<fv_float> centroid;
fv_float length;
size_t label, code,nb_vertex;
FVVertex3D *firstVertex,*secondVertex; // the two vertices
     FVEdge3D(){firstVertex=NULL;secondVertex=NULL;nb_vertex=0;label=0;}
    ~FVEdge3D(){;} 
void setCode2Vertex(size_t val=0)
         {if (firstVertex) firstVertex->code=val;if(secondVertex) secondVertex->code=val;} 
    
private:

};



inline bool isEqual(FVEdge3D *e1, FVEdge3D *e2)
     {
      bool is_equal1 = false, is_equal2 = false;  
      if(e1->firstVertex->label==e2->firstVertex->label) is_equal1=true;   
      if(e1->firstVertex->label==e2->secondVertex->label) is_equal1=true;
      if(e1->secondVertex->label==e2->firstVertex->label) is_equal2=true;
      if(e1->secondVertex->label==e2->secondVertex->label) is_equal2=true;
      return(is_equal1 && is_equal2);
     }
#endif // define _FVEDGE3D

