#include <iostream>
#include "FVLib.h"

#define PI 3.141592653
#define BIG 10e+30

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

#include <tk/stopwatch.hpp>

using tk::Stopwatch;
using tk::Time;

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
double compute_flux(FVMesh2D &m,FVVect<double> &pol,FVVect<FVPoint2D<double> > &V,FVVect<double> &flux,Parameter &para)
{

#ifdef	TIME_FUNCTIONS
	times.functions.compute_flux.timer.start();
#endif

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

return(dt);
}

void    update(FVMesh2D &m,FVVect<double> &pol,FVVect<double> &flux, double dt)
{

#ifdef	TIME_FUNCTIONS
	times.functions.update.timer.start();
#endif

 FVEdge2D *ptr_e;
 m.beginEdge();
 while((ptr_e=m.nextEdge()))
     {
     pol[ptr_e->leftCell->label-1]-=dt*flux[ptr_e->label-1]*ptr_e->length/ptr_e->leftCell->area;
     if(ptr_e->rightCell) pol[ptr_e->rightCell->label-1]+=dt*flux[ptr_e->label-1]*ptr_e->length/ptr_e->rightCell->area;
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
//















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
//FVio pol_file("polution.xml",FVWRITE);
	FVio pol_file( out_fname.c_str() , FVWRITE );
//pol_file.put(pol,time,"polution"); 
//cout<<"computing"<<endl;
while(time<final_time)
//for ( int i = 0 ; i < 10 ; ++i )
{

#ifdef	TIME_ITERATION
		times.iteration.timer.start();
#endif

    dt=compute_flux(m,pol,V,flux,para)*h;
    update(m,pol,flux,dt);
    time+=dt;
//    nbiter++;
//    if(nbiter%nbjump==0)
//        {
//        pol_file.put(pol,time,"polution");    
//        printf("step %d  at time %f \r",(int)nbiter,time); fflush(NULL);
//        }
// 
	{
//		using std::cout;
//		using std::endl;
//		for ( int j = 0 ; j < m.getNbCell() ; ++j )
//			cout
//				<<	'['	<<	j	<<	"]:"	<<	pol[ j ]	<<	endl;
//		getchar();

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
}

pol_file.put(pol,time,"polution"); 


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
