// gmsh2.geo

// variables
H=.1;			// max height of the mesh
W=H*10;			// max width

Y=10;			// number of lines
X=Y*10;			// number of columns

lx = H/Y;		// width of each cell
ly = W/(Y+1);	// height of each cell

// initial points and line
Point(1) = {0.0, 0.0, 0.0, lx};
Point(2) = {0.0, H,	  0.0, lx};
Line(1) = {1, 2};

out[] = Extrude {W, 0, 0} { Line{1}; Layers{{X},{1}}; Recombine; };

// the physical part
Physical Line(1)	 = {1,2} ;		// dirichlet condition on side x=0 x=1
Physical Line(2)	 = {3,4} ;		// neumann condition on side y=0 y=1
Physical Surface(10) = {out[1]} ; 	// code 10 for the cells
