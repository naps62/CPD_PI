#ifndef __FVVERTEX2D_H
#define __FVVERTEX2D_H
#include <vector>
#include "FVPoint2D.h"
#include "FVLib_config.h"
using namespace std;

class FVCell2D;
class FVVertex2D
{
public:
FVPoint2D <double> coord;
size_t label, code, nb_cell,pos_c;
FVCell2D* cell[NB_CELL_PER_VERTEX_2D]; 
     FVVertex2D(){label=0;nb_cell=0;}
    ~FVVertex2D(){;}  
    // iterator
     FVCell2D* beginCell(){pos_c=0;if(pos_c<nb_cell) return(cell[0]);else return(NULL);};
     FVCell2D* nextCell(){if(pos_c<nb_cell) return (cell[pos_c++]);else return(NULL);}    
 
private:

};



inline bool isEqual(FVVertex2D *v1, FVVertex2D *v2)
     {
      if(v1->label==v2->label) return true; else return false;   
     }
#endif // define _FVVERTEX2D
