#include <iostream>
#include "FVLib.h"

#include <fv/cpu/cell.hpp>
#include <fv/cpu/edge.old.hpp>

#include <tk/stopwatch.hpp>

using fv::cpu::Cell;
using fv::cpu::Edge;

using tk::Stopwatch;
using tk::Time;


//
//	BEGIN CONSTANTS
//

#ifdef	TIME_ALL
#define	TIME_MAIN
#define	TIME_ITERATION
#define	TIME_FUNCTIONS
#endif

#if defined	(TIME_MAIN)	\
 || defined (TIME_ITERATION)	\
 || defined (TIME_FUNCTIONS)
#define	TIME
#endif

//
//	END CONSTANTS
//
//
//	BEGIN GLOBALS
//

#ifdef	TIME
struct TimeStats
{
	double min;
	double max;

	friend ostream& operator<<(ostream& out, const TimeStats& ts);

	TimeStats();
};

struct Timer
{
	Stopwatch timer;
	TimeStats miliseconds;
	unsigned count;

	Timer();
};

struct ProgramTimeInfo
{
#ifdef	TIME_MAIN
	struct
	{
		Stopwatch timer;
		double total;
	}
	main;
#endif
#ifdef	TIME_ITERATION
	Timer iteration;
#endif
#ifdef	TIME_FUNCTIONS
	struct
	{
		Timer compute_flux;
		Timer update;
	}
	functions;
#endif
}
times;

TimeStats::TimeStats() :
	min( numeric_limits<double>::max() ),
	max( numeric_limits<double>::min() )
{}

ostream& operator<<(ostream& out, const TimeStats& ts)
{
	out
		<<	"min:   "	<<	ts.min		<<	endl
		<<	"max:	"	<<	ts.max		<<	endl
		;
	return out;
}

Timer::Timer() :
	count(0)
{}
#endif

//
//	END GLOBALS
//


//---------
double compute_flux(
	Edge *edges, unsigned edge_count, Cell *cells,
	double dirichlet)
{

#ifdef	TIME_FUNCTIONS
	times.functions.compute_flux.timer.start();
#endif

	double dt=1.e20;
	FVPoint2D<double> VL,VR;
	double polL,polR,v;
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[e];
		Cell &cell_left = cells[ edge.left ];
		VL.x = cell_left.velocity[0];
		VL.y = cell_left.velocity[1];
		polL = cell_left.polution;
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
			Cell &cell_right = cells[ edge.right ];
			VR.x = cell_right.velocity[0];
			VR.y = cell_right.velocity[1];
			polR = cell_right.polution;
		}
		else
		{
			VR=VL;
			polR= dirichlet;
		} 
		v = ( VL.x + VR.x ) * 0.5 * edge.normal[0]
		  + ( VL.y + VR.y ) * 0.5 * edge.normal[1];
		if (abs(v)*dt>1) dt=1./abs(v);
		edge.flux = ( v < 0 ) ? ( v * polR ) : ( v * polL );
	}

#ifdef	TIME_FUNCTIONS
	times.functions.compute_flux.timer.stop();
	{
		Time partial = times.functions.compute_flux.timer.partial();
		double miliseconds = partial.miliseconds();
		times.functions.compute_flux.miliseconds.min =
			( miliseconds < times.functions.compute_flux.miliseconds.min )
			? miliseconds
			: times.functions.compute_flux.miliseconds.min
			;
		times.functions.compute_flux.miliseconds.max =
			( miliseconds > times.functions.compute_flux.miliseconds.max )
			? miliseconds
			: times.functions.compute_flux.miliseconds.max
			;
		times.functions.compute_flux.count += 1;
	}
#endif

	return dt;
}







void    update(
	Cell *cells,
	unsigned cell_count,
	Edge *edges,
	double dt)
{

#ifdef	TIME_FUNCTIONS
	times.functions.update.timer.start();
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

#ifdef	TIME_FUNCTIONS
	times.functions.update.timer.stop();
	{
		Time partial = times.functions.update.timer.partial();
		double miliseconds = partial.miliseconds();
		times.functions.update.miliseconds.min =
			( miliseconds < times.functions.update.miliseconds.min )
			? miliseconds
			: times.functions.update.miliseconds.min
			;
		times.functions.update.miliseconds.max =
			( miliseconds > times.functions.update.miliseconds.max )
			? miliseconds
			: times.functions.update.miliseconds.max
			;
		times.functions.update.count += 1;
	}
#endif

}















int main(int argc, char *argv[])
{

#ifdef	TIME_MAIN
	times.main.timer.start();
#endif

	string parameter_filename;
	if ( argc > 1 )
		parameter_filename = string( argv[1] );
	else
		parameter_filename = "param.xml";

	string mesh_filename,velo_filename,pol_filename,pol_ini_filename;
	string name;
	// read the parameter
	Parameter para(parameter_filename.c_str());
	mesh_filename=para.getString("MeshName");
	velo_filename=para.getString("VelocityFile");
	pol_ini_filename=para.getString("PoluInitFile");
	string out_fname = para.getString("PoluFile");

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







	// the main loop
	time=0.;nbiter=0;
//	FVio pol_file("polution.omp.xml",FVWRITE);
	FVio pol_file( out_fname.c_str() , FVWRITE );
	//pol_file.put(pol,time,"polution"); 
	//cout<<"computing"<<endl;
	while(time<final_time)
//	for ( int i = 0 ; i < 10 ; ++i )
	{

#ifdef	TIME_ITERATION
		times.iteration.timer.start();
#endif

		dt = compute_flux(
			edges , edge_count , cells , dirichlet ) * h;
		update(
			cells,
			cell_count,
			edges,
			dt);
		time+=dt;

#ifdef	TIME_ITERATION
		times.iteration.timer.stop();
		{
			Time partial = times.iteration.timer.partial();
			double miliseconds = partial.miliseconds();
			times.iteration.miliseconds.min =
				( miliseconds < times.iteration.miliseconds.min )
				? miliseconds
				: times.iteration.miliseconds.min
				;
			times.iteration.miliseconds.max =
				( miliseconds > times.iteration.miliseconds.max )
				? miliseconds
				: times.iteration.miliseconds.max
				;
			times.iteration.count += 1;
		}
#endif

	}

	for ( unsigned c = 0; c < cell_count ; ++c )
		pol[ c ] = cells[ c ].polution;

	pol_file.put(pol,time,"polution"); 


	delete[] cells;
	delete[] edges;

#ifdef	TIME_MAIN
	times.main.timer.stop();
	{
		Time total = times.main.timer.total();
		cout
			<<	"<main>"	<<	endl
			<<	"total: "	<<	total.miliseconds()
							<<	endl
			<<	"</main>"	<<	endl
			;
	}
#endif
#ifdef	TIME_ITERATION
	{
		Time total = times.iteration.timer.total();
		cout
			<<	"<iteration>"	<<	endl
			<<	"total: "	<<	total.miliseconds()	<<	endl
			<<	"min: "		<<	times.iteration.miliseconds.min		<<	endl
			<<	"max: "		<<	times.iteration.miliseconds.max		<<	endl
			<<	"count: "	<<	times.iteration.count	<<	endl
			<<	"</iteration>"	<<	endl
			;
	}
#endif
#ifdef	TIME_FUNCTIONS
	{
		Time total[2];
		total[0] = times.functions.compute_flux.timer.total();
		total[1] = times.functions.update.timer.total();
		cout
			<<	"<functions>"	<<	endl
			<<	"<compute_flux>"	<<	endl
			<<	"total: "	<<	total[0].miliseconds()	<<	endl
			<<	"min: "		<<	times.functions.compute_flux.miliseconds.min	<<	endl
			<<	"max: "		<<	times.functions.compute_flux.miliseconds.max	<<	endl
			<<	"count: "	<<	times.functions.compute_flux.count	<<	endl
			<<	"</compute_flux>"	<<	endl
			<<	"<update>"	<<	endl
			<<	"total: "	<<	total[1].miliseconds()	<<	endl
			<<	"min: "		<<	times.functions.update.miliseconds.min		<<	endl
			<<	"max: "		<<	times.functions.update.miliseconds.max		<<	endl
			<<	"count: "	<<	times.functions.update.count	<<	endl
			<<	"</update>"	<<	endl
			;
	}
#endif


}
