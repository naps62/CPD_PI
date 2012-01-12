// ------  FVLIB_config.h ------
#ifndef _FVLIB_Config 
#define _FVLIB_Config 

#include "MFVLib_config.h"

#define INF_MIN (-1.E+100)
#define SUP_MAX (1.E+100)
#define FVDOUBLE_PRECISION 1.E-17
#define FVPRECISION 12
#define FVCHAMP 20
#define FVCHAMPINT 10

#define NB_CELL_PER_VERTEX_2D 12
#define NB_VERTEX_PER_CELL_2D 9
#define NB_EDGE_PER_CELL_2D 9
#define NB_CELL_PER_VERTEX_3D 60
#define NB_VERTEX_PER_FACE_3D 9
#define NB_EDGE_PER_FACE_3D 9
#define NB_VERTEX_PER_CELL_3D 9  
#define NB_FACE_PER_CELL_3D 9
#define GMSH_NB_NODE_PER_ELEMENT 9
//#define NB_ENTITY_PER_STENCIL 40

#define MINUS_THREE_DIM  2147483648
#define MINUS_TWO_DIM    1073741824
#define MINUS_ONE_DIM     536870912


#include <string>
#include <map>
typedef  std::map<std::string,std::string> StringMap; 



enum FVFile{
           FVNULL     =  0,
           FVOK       ,
           FVREAD     ,
           FVWRITE    ,
           FVENDFILE  ,
           FVNOFILE   ,
           FVWRONGDIM ,
           FVERROR    ,
           VERTEX     ,
           CELL       
};
enum EntityCode{
           NULL_ENTITY=0,
           FVVERTEX1D,
           FVVERTEX2D,
           FVVERTEX3D,
           FVCELL1D,
           FVCELL2D,
           FVCELL3D,
           FVEDGE2D,
           FVEDGE3D,
           FVFACE3D
};
  
enum BaliseCode{
           BadBaliseFormat=0,
           EndXMLFile,
           NoOpenBalise,
           NoCloseBalise,
           OkOpenBalise,
           OkCloseBalise,
           NoAttribute,
           OkAttribute
};
#endif // define _FVLIB_Config
