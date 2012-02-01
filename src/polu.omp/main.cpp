#include <iostream>
#include "FVLib.h"

#include <fv/cpu/edge.hpp>

#define PI 3.141592653
#define BIG 10e+30

using fv::cpu::Edge;

//---------
double compute_flux(FVMesh2D &m,FVVect<double> &pol,FVVect<FVPoint2D<double> > &V,FVVect<double> &flux, Edge *edges, unsigned edge_count, double dirichlet)
{
	double dt=1.e20;
	FVPoint2D<double> VL,VR;
	double polL,polR,v;
//	FVEdge2D *ptr_e;
//	m.beginEdge();
//	while((ptr_e=m.nextEdge()))
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[e];
//		VL=V[ptr_e->leftCell->label-1];
		VL = V[ edge.left ];
//		polL=pol[ptr_e->leftCell->label-1];
		polL = pol[ edge.left ];
//		if(ptr_e->rightCell) 
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
//			VR=V[ptr_e->rightCell->label-1];
			VR = V[ edge.right ];
//			polR=pol[ptr_e->rightCell->label-1];
			polR = pol[ edge.right ];
		}
		else
		{
			VR=VL;
			polR= dirichlet;
		} 
//		v=((VL+VR)*0.5)*(ptr_e->normal); 
		v = ( VL.x + VR.x ) * 0.5 * edge.normal[0]
		  + ( VL.y + VR.y ) * 0.5 * edge.normal[1];
		if (abs(v)*dt>1) dt=1./abs(v);
//		if (v<0) flux[ptr_e->label-1]=v*polR; else flux[ptr_e->label-1]=v*polL;
		flux[ e ] = ( v < 0 ) ? ( v * polR ) : ( v * polL );
	}
	return dt;
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















int main(int argc, char *argv[])
{  
	string parameter_filename="param.xml", mesh_filename,velo_filename,pol_filename,pol_ini_filename;
	string name;
	// read the parameter
	Parameter para(parameter_filename.c_str());
	mesh_filename=para.getString("MeshName");
	velo_filename=para.getString("VelocityFile");
	pol_ini_filename=para.getString("PoluInitFile");

	double dirichlet = para.getDouble("DirichletCondition");

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


	unsigned edge_count = m.getNbEdge();
	Edge *edges = new Edge[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[ e ];
		FVEdge2D *fv_edge = m.getEdge( e );

		edge.flux = flux[ e ];
		edge.length = fv_edge->length;
		edge.normal[0] = fv_edge->normal.x;
		edge.normal[1] = fv_edge->normal.y;
		edge.left = fv_edge->leftCell->label - 1;
		edge.right = ( fv_edge->rightCell )
					 ? fv_edge->rightCell->label - 1
					 : numeric_limits<unsigned>::max();
	}




	// the main loop
	time=0.;nbiter=0;
	FVio pol_file("polution.xml",FVWRITE);
	//pol_file.put(pol,time,"polution"); 
	//cout<<"computing"<<endl;
	//while(time<final_time)
	//    {
	dt=compute_flux(m,pol,V,flux,edges,edge_count,dirichlet)*h;
	//    update(m,pol,flux,dt);
	//    time+=dt;
	//    nbiter++;
	//    if(nbiter%nbjump==0)
	//        {
	//        pol_file.put(pol,time,"polution");    
	//        printf("step %d  at time %f \r",(int)nbiter,time); fflush(NULL);
	//        }
	// 
	//    }

	//pol_file.put(pol,time,"polution"); 

	{
		using std::cout;
		using std::endl;
		cout
			<<	"dt:"	<<	dt	<<	endl;
	}

}
