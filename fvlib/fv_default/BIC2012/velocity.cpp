#include <iostream>
#include "FVLib.h"

 int main(int argc, char *argv[])
{
string parameter_filename="param.xml", mesh_filename,velo_filename,pol_ini_filename,pot_filename;
Parameter para(parameter_filename.c_str());
mesh_filename=para.getString("MeshName");
velo_filename=para.getString("VelocityFile");
pol_ini_filename=para.getString("PoluInitFile");
pot_filename=para.getString("PotentialFile");

FVMesh2D m;
Gmsh mg;
FVCell2D *ptr_c;
FVVertex2D *ptr_v, *ptr_vb;
double dist;
m.read(mesh_filename.c_str());
FVVect<double> pot(m.getNbVertex()), pol(m.getNbCell());
FVVect<FVPoint2D<double> > V(m.getNbCell());
FVPoint2D<double> center;
// compute the potential
for(size_t i=0;i<m.getNbVertex(); i++)
    {
     dist=1.e20;   
     ptr_v=m.getVertex(i);   
     m.beginVertex();
     while((ptr_vb=m.nextVertex()))
        {
         double aux=Norm(ptr_v->coord-ptr_vb->coord);
         if ((dist>aux) && ((ptr_vb->code==2)||(ptr_vb->code==3))) dist=aux;
        }
     pot[i]=-dist;   
    }
// compute the velocity    

FVDenseM<double> M(2);
double aux;
m.beginCell();
while((ptr_c=m.nextCell()))
    {
    FVPoint2D<double> d1;
    d1.x=pot[ptr_c->vertex[1]->label-1]-pot[ptr_c->vertex[0]->label-1];
    d1.y=pot[ptr_c->vertex[2]->label-1]-pot[ptr_c->vertex[0]->label-1];
    aux=ptr_c->vertex[1]->coord.x-ptr_c->vertex[0]->coord.x;
    M.setValue(0,0,aux);
    aux=ptr_c->vertex[1]->coord.y-ptr_c->vertex[0]->coord.y;
    M.setValue(0,1,aux);
    aux=ptr_c->vertex[2]->coord.x-ptr_c->vertex[0]->coord.x;
    M.setValue(1,0,aux);
    aux=ptr_c->vertex[2]->coord.y-ptr_c->vertex[0]->coord.y;
    M.setValue(1,1,aux);
    M.Gauss(d1);
    V[ptr_c->label-1].x=d1.y;V[ptr_c->label-1].y=-d1.x;  
    }
// compute the concentration  
center.x=0.05;center.y=0.3;
m.beginCell();    
while((ptr_c=m.nextCell()))
    {
    pol[ptr_c->label-1]=0;   
    //if(Norm(ptr_c->centroid-center)<0.04)  pol[ptr_c->label-1]=1;
    }
// write in the FVLib format    
FVio velocity_file(velo_filename.c_str(),FVWRITE);
velocity_file.put(V,0.0,"velocity");   
FVio potential_file(pot_filename.c_str(),FVWRITE);
potential_file.put(pot,0.0,"potential");
FVio polu_ini_file(pol_ini_filename.c_str(),FVWRITE);
polu_ini_file.put(pol,0.0,"concentration");
}