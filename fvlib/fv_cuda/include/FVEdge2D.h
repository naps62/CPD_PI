#ifndef __FVEDGE2D_H
#define __FVEDGE2D_H
#include <vector>
#include "FVPoint2D.h"
#include "FVLib_config.h"
using namespace std;

class FVCell2D;
class FVVertex2D;
class FVEdge2D
{
public:
FVPoint2D<fv_float> centroid;
fv_float length;
size_t label, code,nb_vertex,nb_cell;
FVCell2D *leftCell,*rightCell;  // the two cells
FVVertex2D *firstVertex,*secondVertex; // the two vertices
FVPoint2D<fv_float> normal;  // from left to right
     FVEdge2D(){leftCell=NULL;rightCell=NULL;firstVertex=NULL;secondVertex=NULL;label=0;}
    ~FVEdge2D(){;}  
    void setCode2Vertex(size_t val=0)
         {if (firstVertex) firstVertex->code=val;if(secondVertex) secondVertex->code=val;}    
    
private:
};


inline bool isEqual(FVEdge2D *e1, FVEdge2D *e2)
     {
      bool is_equal1 = false, is_equal2 = false;  
      if(e1->firstVertex->label==e2->firstVertex->label) is_equal1=true;   
      if(e1->firstVertex->label==e2->secondVertex->label) is_equal1=true;
      if(e1->secondVertex->label==e2->firstVertex->label) is_equal2=true;
      if(e1->secondVertex->label==e2->secondVertex->label) is_equal2=true;
      return(is_equal1 && is_equal2);
     }
#endif // define _FVEDGE2D

