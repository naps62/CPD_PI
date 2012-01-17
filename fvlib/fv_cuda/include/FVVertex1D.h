#ifndef __FVVERTEX1D_H
#define __FVVERTEX1D_H
#include <vector>
#include "FVPoint1D.h"

class FVCell1D;
class FVVertex1D
{
public:
FVPoint1D<fv_float> coord, normal;
size_t label, code;
FVCell1D *leftCell,*rightCell;
     FVVertex1D(){leftCell=NULL;rightCell=NULL;label=0;coord=0;normal=0;}
    ~FVVertex1D(){;}  
private:

};


inline bool isEqual(FVVertex1D *v1, FVVertex1D *v2)
     {
      if(v1->label==v2->label) return true; else return false;   
     }
#endif // define _FVVERTEX1D

