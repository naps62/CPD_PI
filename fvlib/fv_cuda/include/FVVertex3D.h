#ifndef __FVVERTEX3D_H
#define __FVVERTEX3D_H
#include <vector>
#include "FVPoint3D.h"
#include "FVLib_config.h"
using namespace std;

class FVCell3D;
class FVVertex3D
{
public:
FVPoint3D <double> coord;
size_t label, code, nb_cell,pos_c;
FVCell3D* cell[NB_CELL_PER_VERTEX_3D]; 
     FVVertex3D(){label=0;nb_cell=0;}
    ~FVVertex3D(){;}  
    // iterator
     FVCell3D* beginCell(){pos_c=0;if(pos_c<nb_cell) return(cell[0]);else return(NULL);};
     FVCell3D* nextCell(){if(pos_c<nb_cell) return (cell[pos_c++]);else return(NULL);}        


private:
};



inline bool isEqual(FVVertex3D *v1, FVVertex3D *v2)
     {
      if(v1->label==v2->label) return true; else return false;   
     }
#endif // define _FVVERTEX3D

