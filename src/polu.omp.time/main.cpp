#include <iostream>
#include <limits>

#include <omp.h>

#include "FVLib.h"

#include <fv/cpu/cell.hpp>
#include <fv/cpu/edge.hpp>

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

int tc;

#ifdef	TIME
struct TimeStats
{
	double total;
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
#endif

TimeStats::TimeStats() :
	total(0),
	min( numeric_limits<double>::max() ),
	max( numeric_limits<double>::min() )
{}

ostream& operator<<(ostream& out, const TimeStats& ts)
{
	out
		<<	"total: "	<<	ts.total	<<	endl
		<<	"min:   "	<<	ts.min		<<	endl
		<<	"max:	"	<<	ts.max		<<	endl
		;
	return out;
}

Timer::Timer() :
	count(0)
{}

//
//	END GLOBALS
//

void compute_flux(
	Edge *edges, unsigned edge_count, Cell *cells,
	double dirichlet)
{

#ifdef	TIME_FUNCTIONS
	times.functions.compute_flux.timer.start();
#endif


#pragma omp parallel for num_threads(tc)
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[e];
		Cell &cell_left = cells[ edge.left ];
		double p_left = cell_left.polution;
		double p_right;
		if ( edge.right < numeric_limits<unsigned>::max() )
		{
			Cell &cell_right = cells[ edge.right ];
			p_right = cell_right.polution;
		}
		else
		{
			p_right = dirichlet;
		} 
		edge.flux = ( edge.velocity < 0 )
				  ? ( edge.velocity * p_right )
				  : ( edge.velocity * p_left );
	}


#ifdef	TIME_FUNCTIONS
	times.functions.compute_flux.timer.stop();
	{
		Time total = times.functions.compute_flux.timer.total();
		double miliseconds = total.miliseconds();
		times.functions.compute_flux.miliseconds.total += miliseconds;
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

}

void update(
	Cell *cells,
	unsigned cell_count,
	Edge *edges,
	double dt)
{

#ifdef	TIME_FUNCTIONS
	times.functions.update.timer.start();
#endif


#pragma omp parallel for num_threads(tc)
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
		Time total = times.functions.update.timer.total();
		double miliseconds = total.miliseconds();
		times.functions.update.miliseconds.total += miliseconds;
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
		parameter_filename = string( argv[0] );
	else
		parameter_filename = "param.xml";

	string mesh_filename,velo_filename,pol_filename,pol_ini_filename;
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
	double max_vel = numeric_limits<double>::min();
	Edge *edges = new Edge[ edge_count ];
	for ( unsigned e = 0 ; e < edge_count ; ++e )
	{
		Edge &edge = edges[ e ];
		FVEdge2D *fv_edge = m.getEdge( e );

		edge.flux = flux[ e ];
		edge.length = fv_edge->length;
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


	//
	//	OpenMP init
	//
	tc = omp_get_num_procs() * OMP_FCT_ALL; 





	// the main loop
	time=0.;nbiter=0;
	FVio pol_file("polution.omp.xml",FVWRITE);
	//pol_file.put(pol,time,"polution"); 
	//cout<<"computing"<<endl;
	while(time<final_time)
	{
#ifdef	TIME_ITERATION
		times.iteration.timer.start();
#endif
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
#ifdef	TIME_ITERATION
		times.iteration.timer.stop();
		{
			Time total = times.iteration.timer.total();
			double miliseconds = total.miliseconds();
			times.iteration.miliseconds.total += miliseconds;
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
	cout
		<<	"<main>"	<<	endl
		<<	"total: "	<<	times.main.timer.total().miliseconds()
						<<	endl
		<<	"</main>"	<<	endl
		;
#endif
#ifdef	TIME_ITERATION
	cout
		<<	"<iteration>"	<<	endl
		<<	"total: "	<<	times.iteration.miliseconds.total	<<	endl
		<<	"min: "		<<	times.iteration.miliseconds.min		<<	endl
		<<	"max: "		<<	times.iteration.miliseconds.max		<<	endl
		<<	"count: "	<<	times.iteration.count	<<	endl
		<<	"</iteration>"	<<	endl
		;
#endif
#ifdef	TIME_FUNCTIONS
	cout
		<<	"<functions>"	<<	endl
		<<	"<compute_flux>"	<<	endl
		<<	"total: "	<<	times.functions.compute_flux.miliseconds.total	<<	endl
		<<	"min: "		<<	times.functions.compute_flux.miliseconds.min	<<	endl
		<<	"max: "		<<	times.functions.compute_flux.miliseconds.max	<<	endl
		<<	"count: "	<<	times.functions.compute_flux.count	<<	endl
		<<	"</compute_flux>"	<<	endl
		<<	"<update>"	<<	endl
		<<	"total: "	<<	times.functions.update.miliseconds.total	<<	endl
		<<	"min: "		<<	times.functions.update.miliseconds.min		<<	endl
		<<	"max: "		<<	times.functions.update.miliseconds.max		<<	endl
		<<	"count: "	<<	times.functions.update.count	<<	endl
		<<	"</update>"	<<	endl
		;
#endif
}
