#include <iostream>
#include "FVLib.h"

#define PI 3.141592653
#define BIG 10e+30


//---------
double compute_flux(FVMesh2D &m,FVVect<double> &pol,FVVect<FVPoint2D<double> > &V,FVVect<double> &flux,Parameter &para)
{
double dt=1.e20;
 FVPoint2D<double> VL,VR;
 double polL,polR,v;
 FVEdge2D *ptr_e;
 m.beginEdge();
 while((ptr_e=m.nextEdge()))
     {
      VL=V[ptr_e->leftCell->label-1];
      polL=pol[ptr_e->leftCell->label-1];
     if(ptr_e->rightCell) 
        {
        VR=V[ptr_e->rightCell->label-1];
        polR=pol[ptr_e->rightCell->label-1];
        }
     else
        {
        VR=VL;
        polR=para.getDouble("DirichletCondition");
        } 
     v=((VL+VR)*0.5)*(ptr_e->normal); 
     if (abs(v)*dt>1) dt=1./abs(v);
     if (v<0) flux[ptr_e->label-1]=v*polR; else flux[ptr_e->label-1]=v*polL;
     }
return(dt);
}

void    update(FVMesh2D &m,FVVect<double> &pol,FVVect<double> &flux, double dt)
{
 FVEdge2D *ptr_e;
 m.beginEdge();
 while((ptr_e=m.nextEdge()))
     {
     pol[ptr_e->leftCell->label-1]-=dt*flux[ptr_e->label-1]*ptr_e->length/ptr_e->leftCell->area;
     if(ptr_e->rightCell) pol[ptr_e->rightCell->label-1]+=dt*flux[ptr_e->label-1]*ptr_e->length/ptr_e->rightCell->area;
     }
}    
//















int main()
{  
string parameter_filename="param.xml", mesh_filename,velo_filename,pol_filename,pol_ini_filename;
string name;
// read the parameter
Parameter para(parameter_filename.c_str());
mesh_filename=para.getString("MeshName");
velo_filename=para.getString("VelocityFile");
pol_ini_filename=para.getString("PoluInitFile");
double time,final_time,dt,h,S;
size_t nbiter,nbjump;
FVMesh2D m;
FVCell2D *ptr_c;
FVEdge2D *ptr_e;
// read the mesh
m.read(mesh_filename.c_str());
FVVect<double> pol(m.getNbCell()),flux(m.getNbEdge());
FVVect<FVPoint2D<double> > V(m.getNbCell());
// read the  data  and initialisation
FVio velocity_file(velo_filename.c_str(),FVREAD);
velocity_file.get(V,time,name);
FVio polu_ini_file(pol_ini_filename.c_str(),FVREAD);
polu_ini_file.get(pol,time,name);
final_time=para.getDouble("FinalTime");
nbjump=para.getInteger("NbJump");
// compute the Mesh parameter
h=1.e20;
m.beginCell();
while((ptr_c=m.nextCell()))
    {
    S=ptr_c->area;
    ptr_c->beginEdge();
    while((ptr_e=ptr_c->nextEdge()))
        {
         if(h*ptr_e->length>S) h=S/ptr_e->length;
        }
    }
// the main loop
time=0.;nbiter=0;
FVio pol_file("polution.xml",FVWRITE);
pol_file.put(pol,time,"polution"); 
cout<<"computing"<<endl;
while(time<final_time)
    {
    dt=compute_flux(m,pol,V,flux,para)*h;
    update(m,pol,flux,dt);
    time+=dt;
    nbiter++;
    if(nbiter%nbjump==0)
        {
        pol_file.put(pol,time,"polution");    
        printf("step %d  at time %f \r",(int)nbiter,time); fflush(NULL);
        }
 
    }

pol_file.put(pol,time,"polution"); 
}

