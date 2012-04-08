#include <iostream>
#include "FVLib.h"


#ifdef PROFILE_LIMITED
unsigned mliters;
#endif


double compute_flux(FVMesh2D &m,FVVect<double> &pol,FVVect<FVPoint2D<double> > &V,FVVect<double> &flux,Parameter &para)
{

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	#endif
		PROFILE_START();
#endif

	double dt = 1.e20;
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

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	{
	#endif
		PROFILE_STOP();
		PROFILE_RETRIEVE_CF();
	#ifdef PROFILE_WARMUP
	}
	#endif
#endif

	return(dt);
}

void    update(FVMesh2D &m,FVVect<double> &pol,FVVect<double> &flux, double dt)
{

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	#endif
		PROFILE_START();
#endif

	FVEdge2D *ptr_e;
	m.beginEdge();
	while((ptr_e=m.nextEdge()))
	{
		pol[ptr_e->leftCell->label-1]-=dt*flux[ptr_e->label-1]*ptr_e->length/ptr_e->leftCell->area;
		if(ptr_e->rightCell) pol[ptr_e->rightCell->label-1]+=dt*flux[ptr_e->label-1]*ptr_e->length/ptr_e->rightCell->area;
	}

#ifdef PROFILE
	#ifdef PROFILE_WARMUP
	if ( mliters > PROFILE_WARMUP )
	{
	#endif
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP();
	#ifdef PROFILE_WARMUP
	}
	#endif
#endif

}    















int main(int argc, char *argv[])
{
#ifdef PROFILE
	PROFILE_INIT();
#endif

	string parameter_filename;

	if ( argc > 1 )
		parameter_filename = argv[1];
	else
		parameter_filename = "param.xml";

	string mesh_filename,velo_filename,pol_filename,pol_ini_filename;
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
#ifdef PROFILE_LIMITED
	for ( mliters = 0 ; mliters < PROFILE_LIMITED ; ++mliters )
#else
	for ( ; time < final_time ; time += dt )
#endif
	{
		dt=compute_flux(m,pol,V,flux,para)*h;
		update(m,pol,flux,dt);
	}

	FVio pol_file("polution.xml",FVWRITE);
	pol_file.put(pol,time,"polution"); 

#ifdef PROFILE
	PROFILE_OUTPUT();
	PROFILE_CLEANUP();
#endif
}
