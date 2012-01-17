#include <iostream>
#include "FVLib.h"

//	BEGIN TYPES

struct Parameters
{
	struct
	{
		string mesh;
		string velocity;
		string potential;
		struct
		{
			string initial;
		} polution;
	} filenames;
};

//	END TYPES

/*
	Read parameter file
*/
void read_parameters(
	const char* filename,
	Parameters& data)
{
	Parameter para( filename );

	data.filenames.mesh = para.getString("MeshName");
	data.filenames.velocity = para.getString("VelocityFile");
	data.filenames.polution.initial = para.getString("PoluInitFile");
	data.filenames.potential = para.getString("PotentialFile");
}

/*
	Função madre
	@param	parameter_filename
*/
int main(int argc, char *argv[])
{
	Parameters data;

	if ( argc > 1 )
		read_parameters( argv[1] , data );
	else
		read_parameters( string("param.xml").c_str() , data );

	/*
	string parameter_filename="param.xml";
	string mesh_filename,velo_filename,pol_ini_filename,pot_filename;
	Parameter para(parameter_filename.c_str());
	mesh_filename=para.getString("MeshName");
	velo_filename=para.getString("VelocityFile");
	pol_ini_filename=para.getString("PoluInitFile");
	pot_filename=para.getString("PotentialFile");
	*/

	//FVMesh2D m;
	FVMesh2D mesh;
	Gmsh mg;
	FVCell2D *ptr_c;
	FVVertex2D *ptr_v, *ptr_vb;
	fv_float dist;
	//m.read(mesh_filename.c_str());
	mesh.read( data.filenames.mesh.c_str() );
	FVVect<fv_float> pot( mesh.getNbVertex() );
	FVVect<fv_float> pol( mesh.getNbCell() );
	FVVect<FVPoint2D<fv_float> > V( mesh.getNbCell() );
	FVPoint2D<fv_float> center;

	

	// compute the potential
	for (size_t i=0; i < mesh.getNbVertex(); ++i)
	{
		dist = 1.e20;   
		ptr_v = mesh.getVertex(i);
		mesh.beginVertex();
		while ( ( ptr_vb = mesh.nextVertex() ) )
		{
			fv_float aux = Norm( ptr_v->coord-ptr_vb->coord );
			if ( ( dist > aux )
			&& (  ( ptr_vb->code == 2 )
			   || ( ptr_vb->code == 3 )
			   )
			)
				dist = aux;
		}
		pot[i] = ( - dist );   
	}
	// compute the velocity    

	FVDenseM<fv_float> M(2);
	fv_float aux;
	mesh.beginCell();
	while ( ( ptr_c = mesh.nextCell() ) )
	{
		FVPoint2D<fv_float> d1;
		d1.x = pot[ ptr_c->vertex[1]->label - 1 ] - pot[ ptr_c->vertex[0]->label - 1 ];
		d1.y = pot[ ptr_c->vertex[2]->label - 1 ] - pot[ ptr_c->vertex[0]->label - 1 ];
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
	mesh.beginCell();    
	while((ptr_c=mesh.nextCell()))
	{
		pol[ptr_c->label-1]=0;   
		//if(Norm(ptr_c->centroid-center)<0.04)  pol[ptr_c->label-1]=1;
	}
	// write in the FVLib format    
	FVio velocity_file( data.filenames.velocity.c_str() ,FVWRITE );
	velocity_file.put(V,0.0,"velocity");   
	FVio potential_file( data.filenames.potential.c_str(), FVWRITE );
	potential_file.put(pot,0.0,"potential");
	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVWRITE );
	polu_ini_file.put(pol,0.0,"concentration");
}

