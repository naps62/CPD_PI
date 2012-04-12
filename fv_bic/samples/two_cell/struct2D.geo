// gmsh2.geo
// definition and variable
MINUS_THREE_DIM = 2147483648;
MINUS_TWO_DIM   = 1073741824;
MINUS_ONE_DIM   =  536870912;
L=1.;
N=2;  // the number of cell
lc = 1/N;
// geometrical part
Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {0.0, L, 0.0, lc} ;


Line(1) = {1,2} ;

out[] = Extrude{L,0,0}{Line{1};Layers{{N},{1}}; Recombine;};
Printf("Top line=%g",out[0]);
Printf("Surface=%g",out[1]);
Printf("Side line=%g",out[2]);
Printf("Side line=%g",out[3]);
//Line Loop(5) = {4,1,-2,3} ;
//Plane Surface(6) = {5} ;

//   the physical part

//Physical Point(13) = {1,2} ;
Physical Line(1) = {1,2} ; // dirichlet condition on side x=0 x=1
Physical Line(2) = {3,4} ; // neumann condition on side y=0 y=1
Physical Surface(10) = {out[1]} ;  // code 10 for the cells

//Physical Line(18+MINUS_ONE_DIM) = {4} ;
//Physical Surface(10) = {6} ;
//Physical Surface(12+MINUS_ONE_DIM) = {6} ;

