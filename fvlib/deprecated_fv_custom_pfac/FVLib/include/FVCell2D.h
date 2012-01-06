#ifndef __FVCELL2D_H
#define __FVCELL2D_H
#include <vector>
#include "FVPoint2D.h"
#include "FVEdge2D.h"
#include "FVLib_config.h"
using namespace std;

class FVVertex2D;
class FVEdge2D;
class FVCell2D
{
public:
FVPoint2D<double> centroid;
size_t label,code,nb_vertex,nb_edge;
size_t pos_e,pos_v;
double perimeter,area;
FVVertex2D* vertex[NB_VERTEX_PER_CELL_2D] ; // the  vertices
FVEdge2D* edge[NB_EDGE_PER_CELL_2D];     // the edge
FVPoint2D<double>  cell2edge[NB_EDGE_PER_CELL_2D]; // normal exterior for each edge


     FVCell2D(){nb_vertex=0;nb_edge=0;label=0;}
    ~FVCell2D(){;}  


     FVVertex2D* beginVertex(){pos_v=0;if(pos_v<nb_vertex) return(vertex[0]);else return(NULL);}
     FVVertex2D* nextVertex(){if(pos_v<nb_vertex) return(vertex[pos_v++]);else return(NULL);}  
     FVEdge2D* beginEdge(){pos_e=0;if(pos_e<nb_edge) return(edge[0]);else return(NULL);};
     FVEdge2D* nextEdge(){if(pos_e<nb_edge) return(edge[pos_e++]);else return(NULL);} 
     FVPoint2D<double> getCell2Edge(){return cell2edge[pos_e-1];}
     void setCode2Edge(size_t val=0)
         {for(size_t i=0;i<nb_edge;i++) 
          if(edge[i]) edge[i]->code=val;}
     void setCode2Vertex(size_t val=0)
         { 
             for(size_t i=0;i<nb_edge;i++) 
                 if(edge[i]) 
                 {edge[i]->firstVertex->code=val;edge[i]->secondVertex->code=val;}
             
         }    
private:

};

inline bool isEqual(FVCell2D *c1, FVCell2D *c2)
     {
      bool is_equal,is_equal_total;
      if(c1->nb_edge!=c2->nb_edge) return false;
      is_equal_total=true;
      for(size_t i=0;i<c1->nb_edge;i++)
         {
          is_equal=false;
          for(size_t j=0;j<c2->nb_edge;j++)
             if(c1->edge[i]==c2->edge[j]) is_equal=true; 
          is_equal_total =is_equal && is_equal_total;   
         }
       return(is_equal_total);  
     }

#endif // define _FVCELL2D
