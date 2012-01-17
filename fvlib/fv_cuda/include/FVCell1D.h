#ifndef __FVCELL1D_H
#define __FVCELL1D_H


class FVVertex1D;
class FVCell1D
{
public:
FVPoint1D<fv_float> centroid;
fv_float length;
size_t label,code,nb_vertex;
FVVertex1D *firstVertex,*secondVertex; // the two vertices
FVPoint1D<fv_float> first_normal,second_normal; // normal exterior


     FVCell1D(){firstVertex=NULL;secondVertex=NULL;label=0;_pos_v=0;}
    ~FVCell1D(){;}  
    
     FVVertex1D* beginVertex(){_pos_v=0;return(firstVertex);}
     FVVertex1D* nextVertex(){
                              _pos_v++;
                              if(_pos_v==1) return(firstVertex);
                              if(_pos_v==2) return(secondVertex);
                              return(NULL);}  
    void setCode2Vertex(size_t val=0)
         {if (firstVertex) firstVertex->code=val;if(secondVertex) secondVertex->code=val;}
private:
size_t _pos_v;
};

inline bool isEqual(FVCell1D *c1, FVCell1D *c2)
     {
      bool is_equal1 = false, is_equal2 = false;  
      if(c1->firstVertex->label==c2->firstVertex->label) is_equal1=true;   
      if(c1->firstVertex->label==c2->secondVertex->label) is_equal1=true;
      if(c1->secondVertex->label==c2->firstVertex->label) is_equal2=true;
      if(c1->secondVertex->label==c2->secondVertex->label) is_equal2=true;
      return(is_equal1 && is_equal2);
     }

#endif // define _FVCELL1D
