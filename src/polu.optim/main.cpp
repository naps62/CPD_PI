#if   defined (PAPI_INSTRUCTIONS_TOTAL)	\
 ||   defined (PAPI_INSTRUCTIONS_LOADS) \
 ||   defined (PAPI_INSTRUCTIONS_STORES) \
 ||   defined (PAPI_INSTRUCTIONS_BRANCHS) \
 ||   defined (PAPI_INSTRUCTIONS_FPS) \
 ||   defined (PAPI_INSTRUCTIONS_INTS)
#define PAPI_INSTRUCTIONS
#endif//PAPI_INSTRUCTIONS_*

#if   defined (PAPI_INSTRUCTIONS)
#define PAPIFU
#endif


#include <iostream>
#include <limits>
#include "FVLib.h"

#include <fv/cpu/cell.hpp>
using fv::cpu::Cell;
#include <fv/cpu/edge.hpp>
using fv::cpu::Edge;


#if   defined (PAPIFU)
#if   defined (PAPI_INSTRUCTIONS)
#if   defined (PAPI_INSTRUCTIONS_TOTAL)
#include <papi/total_instructions_preset_counter.hpp>
papi::TotalInstructionsPresetCounter *p;
long long int tot_ins;
#elif defined (PAPI_INSTRUCTIONS_LOADS)
#include <papi/load_instructions_preset_counter.hpp>
papi::LoadInstructionsPresetCounter *p;
long long int ld_ins;
#elif defined (PAPI_INSTRUCTIONS_STORES)
#include <papi/store_instructions_preset_counter.hpp>
papi::StoreInstructionsPresetCounter *p;
long long int sr_ins;
#elif defined (PAPI_INSTRUCTIONS_BRANCHS)
#include <papi/branch_instructions_preset_counter.hpp>
papi::BranchInstructionsPresetCounter *p;
long long int br_ins;
#elif defined (PAPI_INSTRUCTIONS_FPS)
#include <papi/fp_instructions_preset_counter.hpp>
papi::FloatingPointInstructionsPresetCounter *p;
long long int fp_ins;
#elif defined (PAPI_INSTRUCTIONS_INTS)
#include <papi/int_instructions_preset_counter.hpp>
papi::IntegerInstructionsPresetCounter *p;
long long int int_ins;
#endif//PAPI_INSTRUCTIONS_*
long long int ctl_ins;
#endif//PAPI_INSTRUCTIONS
#endif//PAPI



void compute_flux(
	Edge *edges,
	unsigned edge_count,
	Cell *cells,
	double dirichlet)
{
	double polution_left;
	double polution_right;

#if   defined (PAPIFU)
	p->start();
#endif

	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[e];
		Cell &cell_left = cells[ edge.left ];
		polution_left = cell_left.polution;
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
			Cell &cell_right = cells[ edge.right ];
			polution_right = cell_right.polution;
		}
		else
		{
			polution_right= dirichlet;
		} 
		edge.flux = ( edge.velocity < 0 )
				  ? ( edge.velocity * polution_right )
				  : ( edge.velocity * polution_left );
	}

#if   defined (PAPIFU)
	p->stop();
#if   defined (PAPI_INSTRUCTIONS)
	long long int inst = p->instructions() - ctl_ins;
#if   defined (PAPI_INSTRUCTIONS_TOTAL)
	tot_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_LOADS)
	ld_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_STORES)
	sr_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_BRANCHS)
	br_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_FPS)
	fp_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_INTS)
	int_ins += inst;
#endif//PAPI_INSTRUCTIONS_*
#endif//PAPI_INSTRUCTIONS
#endif//PAPI
}




void    update(
	Cell *cells,
	unsigned cell_count,
	Edge *edges,
	double dt)
{

#if   defined  PAPIFU
	p->start();
#endif

	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		Cell &cell = cells[ c ];
		double cdp = 0;

		for ( unsigned e = 0 ; e < cell.edge_count ; ++e )
		{
			Edge &edge = edges[ cell.edges[ e ] ];
			double edp = dt * edge.flux * edge.length / cell.area;
			if ( c == edge.left )
				cdp -= edp;
			else
				cdp += edp;
		}

		cell.polution += cdp;
	}

#if   defined (PAPIFU)
	p->stop();
#if   defined (PAPI_INSTRUCTIONS)
	long long int inst = p->instructions() - ctl_ins;
#if   defined (PAPI_INSTRUCTIONS_TOTAL)
	tot_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_LOADS)
	ld_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_STORES)
	sr_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_BRANCHS)
	br_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_FPS)
	fp_ins += inst;
#elif defined (PAPI_INSTRUCTIONS_INTS)
	int_ins += inst;
#endif//PAPI_INSTRUCTIONS_*
#endif//PAPI_INSTRUCTIONS
#endif//PAPI

}    















int main(int argc, char *argv[])
{  
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
	string pol_fname = para.getString( "PoluFile" );

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
	double max_vel = numeric_limits<double>::min();
	Edge *edges = new Edge[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[ e ];
		FVEdge2D *fv_edge = m.getEdge( e );

		edge.flux = flux[ e ];
		edge.length = fv_edge->length;
//		edge.normal[0] = fv_edge->normal.x;
//		edge.normal[1] = fv_edge->normal.y;
		edge.left = fv_edge->leftCell->label - 1;
		edge.right = ( fv_edge->rightCell )
					 ? fv_edge->rightCell->label - 1
					 : numeric_limits<unsigned>::max();

		double normal[2];
		normal[0] = fv_edge->normal.x;
		normal[1] = fv_edge->normal.y;
		double v_left[2];
		v_left[0] = V[ edge.left ].x;
		v_left[1] = V[ edge.left ].y;
		double v_right[2];
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
			v_right[0] = V[ edge.right ].x;
			v_right[1] = V[ edge.right ].y;
		}
		else
		{
			v_right[0] = v_left[0];
			v_right[1] = v_left[1];
		}

		edge.velocity = ( v_left[0] + v_right[0] ) * 0.5 * normal[0]
					  + ( v_left[1] + v_right[1] ) * 0.5 * normal[1];

		max_vel = ( abs( edge.velocity ) > max_vel )
				? abs( edge.velocity )
				: max_vel
				;
	}


	dt = h / max_vel;



	unsigned cell_count = m.getNbCell();
	Cell *cells = new Cell[ cell_count ];
	for ( unsigned c = 0 ; c < cell_count ; ++c )
	{
		Cell &cell = cells[ c ];
		FVCell2D *fv_cell = m.getCell( c );

		cell.velocity[0] = V[ c ].x;
		cell.velocity[1] = V[ c ].y;
		cell.polution = pol[ c ];
		cell.area = fv_cell->area;
		cell.init( fv_cell->nb_edge );
		for ( unsigned e = 0 ; e < cell.edge_count ; ++e )
			cell.edges[ e ] = fv_cell->edge[ e ]->label - 1;
	}



#if   defined (PAPIFU)
	papi::init();
#if   defined (PAPI_INSTRUCTIONS)
#if   defined (PAPI_INSTRUCTIONS_TOTAL)
	p = new papi::TotalInstructionsPresetCounter();
	tot_ins = 0;
#elif defined (PAPI_INSTRUCTIONS_LOADS)
	p = new papi::LoadInstructionsPresetCounter();
	ld_ins = 0;
#elif defined (PAPI_INSTRUCTIONS_STORES)
	p = new papi::StoreInstructionsPresetCounter();
	sr_ins = 0;
#elif defined (PAPI_INSTRUCTIONS_BRANCHS)
	p = new papi::BranchInstructionsPresetCounter();
	br_ins = 0;
#elif defined (PAPI_INSTRUCTIONS_FPS)
	p = new papi::FloatingPointInstructionsPresetCounter();
	fp_ins = 0;
#elif defined (PAPI_INSTRUCTIONS_INTS)
	p = new papi::IntegerInstructionsPresetCounter();
	int_ins = 0;
#endif//PAPI_INSTRUCTIONS_*
	p->start();
	p->stop();
	ctl_ins = p->instructions();
#endif//PAPI_INSTRUCTIONS
#endif//PAPI




	// the main loop
	time=0.;nbiter=0;
	FVio pol_file( pol_fname.c_str() ,FVWRITE);
	//FVio pol_file("polution.omp.xml",FVWRITE);
	//pol_file.put(pol,time,"polution"); 
	//cout<<"computing"<<endl;
	while(time<final_time)
//	for ( int i = 0 ; i < 10 ; ++i )
	{
//		cout << "--[" << time << "]--" << endl;

//		dt = compute_flux(
		compute_flux(
			edges,
			edge_count,
			cells,
			dirichlet)
		;
//		* h;

		update(
			cells,
			cell_count,
			edges,
//			edge_count,
			dt);
		time += dt;
	//    nbiter++;
	//    if(nbiter%nbjump==0)
	//        {
	//        pol_file.put(pol,time,"polution");    
	//        printf("step %d  at time %f \r",(int)nbiter,time); fflush(NULL);
	//        }
	// 
		{
//			using std::cout;
//			using std::endl;
//			for ( int j = 0 ; j < cell_count ; ++j )
//				cout
//					<<	'['	<<	j	<<	"]:"	<<	cells[j].polution	<<	endl;
//			getchar();
		}
	}

	for ( unsigned c = 0; c < cell_count ; ++c )
		pol[ c ] = cells[ c ].polution;

	pol_file.put(pol,time,"polution"); 

#if   defined (PAPIFU)
	delete p;
	papi::shutdown();
	cout
#if   defined (PAPI_INSTRUCTIONS)
#if   defined (PAPI_INSTRUCTIONS_TOTAL)
		<<	tot_ins
#elif defined (PAPI_INSTRUCTIONS_LOADS)
		<<	ld_ins
#elif defined (PAPI_INSTRUCTIONS_STORES)
		<<	sr_ins
#elif defined (PAPI_INSTRUCTIONS_BRANCHS)
		<<	br_ins
#elif defined (PAPI_INSTRUCTIONS_FPS)
		<<	fp_ins
#elif defined (PAPI_INSTRUCTIONS_INTS)
		<<	int_ins
#endif//PAPI_INSTRUCTIONS_*
#endif//PAPI_INSTRUCTIONS
		<<	endl;
#endif//PAPI

}