#include "FVGaussPoint.h"
//
//
//  class for 1D Gauss Point
//
//
FVPoint2D<double> FVGaussPoint1D::getPoint(size_t order, size_t no=1)
{
FVPoint2D<double> coord;    
switch(order)
    {        
    case 0:
    case 1:
    coord.x=0.5;coord.y=0.5;    
    return(coord);break;
    case 2:
    case 3:
        
    switch(no)
        {
        case 1:
        coord.x=(1.-1./sqrt(3.))*0.5; coord.y=1.-coord.x;
        return(coord);break;
        case 2:
        coord.y=(1.-1./sqrt(3.))*0.5; coord.x=1.-coord.y;            
        return(coord);break;
        default:coord.x=coord.y=0.;return(coord);
        }
    break;
    case 4:
    case 5:
    switch(no)
        {
        case 1:
        coord.x=(1.-sqrt(3./5))*0.5; coord.y=1.-coord.x;            
        return(coord);break;
        case 2:
        coord.x=0.5;coord.y=0.5;       
        return(coord);break;
        case 3:
        coord.y=(1.-sqrt(3./5))*0.5; coord.x=1.-coord.y;              
        return(coord);break;
        default:coord.x=coord.y=0.;return(coord);
        }        
    break;
    default:coord.x=coord.y=0.;return(coord);
    }       
}

double FVGaussPoint1D::getWeight(size_t order, size_t no=1)
{
switch(order)
    {        
    case 0:
    case 1:
    return(1);break;
    case 2:
    case 3:
        
    switch(no)
        {
        case 1:
        return(0.5);break;
        case 2:          
        return(0.5);break;
        default:return(0);
        }
    break;
    case 4:
    case 5:
    switch(no)
        {
        case 1:         
        return(5./18);break;
        case 2:     
        return(8./18);break;
        case 3:            
        return(5./18);break;
        default:return(0);
        }        
    break;
    default:return(0);
    }          
}

size_t FVGaussPoint1D::getNbPoint(size_t order)
{
switch(order)
    {        
    case 0:
    case 1:
    return(1);break;
    case 2:
    case 3:
    return(2);break;
    case 4:
    case 5:
    return(3);break;
    default:return(0);
    }     
}

//
//
//  class for 2D Gauss Point
//
//

FVPoint3D<double> FVGaussPoint2D::getPoint(size_t order, size_t no=1)
{
FVPoint3D<double> coord;   
double a1=(6.-sqrt(15.))/21;
double a2=(6.+sqrt(15.))/21;
switch(order)
    {        
    case 0:
    case 1:
    coord.x=1/3.;coord.y=1/3.;coord.z=1/3.;   
    return(coord);break;
    
    case 2:
    case 3:
    switch(no)
        {
        case 1:
        coord.x=1./3;coord.y=1./3;coord.z=1./3;
        return(coord);break;
        case 2:
        coord.x=3./5;coord.y=1./5;coord.z=1./5;          
        return(coord);break;
        case 3:
        coord.x=1./5;coord.y=3./5;coord.z=1./5;          
        return(coord);break;
        case 4:
        coord.x=1./5;coord.y=1./5;coord.z=3./5;          
        return(coord);break;        
        default:coord.x=coord.y=coord.z=0.;return(coord);
        }
    break;
    
    case 4:
    case 5:
    switch(no)
        {
        case 1:
        coord.x=1./3;coord.y=1./3;coord.z=1./3;  
        return(coord);break;
        case 2:
        coord.x=a1;coord.y=a1;coord.z=1-2*a1;      
        return(coord);break;
        case 3:
        coord.x=a1;coord.y=1-2*a1;coord.z=a1;         
        return(coord);break;
        case 4:
        coord.x=1-2*a1;coord.y=a1;coord.z=a1;     
        return(coord);break;
        case 5:
        coord.x=a2;coord.y=a2;coord.z=1-2*a2;              
        return(coord);break;
        case 6:
        coord.x=a2;coord.y=1-2*a2;coord.z=a2;     
        return(coord);break;
        case 7:
        coord.x=1-2*a2;coord.y=a2;coord.z=a2;              
        return(coord);break;                    
        default:coord.x=coord.y=coord.z=0.;return(coord);
        }        
    break;
    default:coord.x=coord.y=coord.z=0.;return(coord);
    }       
}
double FVGaussPoint2D::getWeight(size_t order, size_t no=1)
{
switch(order)
    {        
    case 0:
    case 1:
    return(1);break;
    case 2:
    case 3:
        
    switch(no)
        {
        case 1:
        return(-27./48);break;
        case 2:
        case 3:
        case 4:
        return(25./48);break;
        default:return(0);
        }
    break;
    case 4:
    case 5:
    switch(no)
        {
        case 1:         
        return(9./40);break;
        case 2: 
        case 3:
        case 4:
        return((155.-sqrt(15.))/1200);break;
        case 5:
        case 6:
        case 7:
        return((155.+sqrt(15.))/1200);break;
        default:return(0);
        }        
    break;
    default:return(0);
    }          
}
size_t FVGaussPoint2D::getNbPoint(size_t order)
{
switch(order)
    {        
    case 0:
    case 1:
    return(1);break;
    case 2:
    case 3:
    return(4);break;
    case 4:
    case 5:
    return(7);break;
    default:return(0);
    }     
}


//
//
//  class for 3D Gauss Point
//
//

FVPoint4D<double> FVGaussPoint3D::getPoint(size_t order, size_t no=1)
{
FVPoint4D<double> coord;  
double a=(10.-2*sqrt(15.))/40,a1=(7.-sqrt(15.))/34,a2=(7.+sqrt(15.))/34;
switch(order)
    {        
    case 0:
    case 1:
    coord.x=1./4.,coord.y=1./4;coord.z=1./4.;coord.t=1./4.;    
    return(coord);break;
    case 2:
    case 3:
        
    switch(no)
        {
        case 1:
        coord.x=1./4.,coord.y=1./4;coord.z=1./4.;coord.t=1./4.;              
        return(coord);break;
        case 2:
        coord.x=1./2.,coord.y=1./6;coord.z=1./6.;coord.t=1./6.;   
        return(coord);break;
        case 3:
        coord.x=1./6.,coord.y=1./2;coord.z=1./6.;coord.t=1./6.;  
        return(coord);break;
        case 4:
        coord.x=1./6.,coord.y=1./6;coord.z=1./2.;coord.t=1./6.;   
        return(coord);break;  
        case 5:
        coord.x=1./6.,coord.y=1./6;coord.z=1./6.;coord.t=1./2.;   
        return(coord);break;          
        
        default:coord.x=coord.y=coord.z=coord.t=0.;return(coord);
        }
    break;
    case 4:
    case 5:
    switch(no)
        {
        case 1:
        coord.x=1./4.,coord.y=1./4;coord.z=1./4.;coord.t=1./4.; 
        return(coord);break;
        case 2:
        coord.x=a1,coord.y=a1;coord.z=a1;coord.t=1-3*a1;          
        return(coord);break;        
        case 3:
        coord.x=a1,coord.y=a1;coord.z=1-3*a1;coord.t=a1;  
        return(coord);break;
        case 4:
        coord.x=a1,coord.y=1-3*a1;coord.z=a1;coord.t=a1;           
        return(coord);break;  
        case 5:
        coord.x=1-3*a1,coord.y=a1;coord.z=a1;coord.t=a1;            
        return(coord);break; 
        case 6:
        coord.x=a2,coord.y=a2;coord.z=a2;coord.t=1-3*a2;  
        return(coord);break;
        case 7:
        coord.x=a2,coord.y=a2;coord.z=1-3*a2;coord.t=a2;            
        return(coord);break;       
        case 8:
        coord.x=a2,coord.y=1-3*a2;coord.z=a2;coord.t=a2;  
        return(coord);break;
        case 9:
        coord.x=1-3*a2,coord.y=a2;coord.z=a2;coord.t=a2;            
        return(coord);break;  
        case 10:
        coord.x=a,coord.y=a;coord.z=0.5-a;coord.t=0.5-a;             
        return(coord);break; 
        case 11:
        coord.x=a,coord.y=0.5-a;coord.z=a;coord.t=0.5-a;   
        return(coord);break;
        case 12:
        coord.x=0.5-a,coord.y=a;coord.z=a;coord.t=0.5-a;             
        return(coord);break;          
        case 13:
        coord.x=a,coord.y=0.5-a;coord.z=0.5-a;coord.t=a;   
        return(coord);break;
        case 14:
        coord.x=0.5-a,coord.y=a;coord.z=0.5-a;coord.t=a;            
        return(coord);break;  
        case 15:
        coord.x=0.5-a,coord.y=0.5-a;coord.z=a;coord.t=a;            
        return(coord);break;         
        default:coord.x=coord.y=coord.z=coord.t=0.;return(coord);
        }        
    break;
    default:coord.x=coord.y=coord.z=coord.t=0.;return(coord);
    }       
}
double FVGaussPoint3D::getWeight(size_t order, size_t no=1)
{
double  w=10./189 ;   
double  w1=(2665.+14*sqrt(15.))/37800;
double  w2=(2665.-14*sqrt(15.))/37800;
switch(order)
    {        
    case 0:
    case 1:
    return(1);break;
    
    case 2:
    case 3:
    switch(no)
        {
        case 1:
        return(-4./5);break;
        case 2:          
        return(9./20);break;
        case 3:
        return(9./20);break;
        case 4:          
        return(9./20);break;        
        case 5:
        return(9./20);break;
        default:return(0);
        }
    break;
    case 4:
    case 5:
    switch(no)
        {
        case 1:
        return(16./135);break;
        case 2:          
        return(w1);break;
        case 3:
        return(w1);break;
        case 4:          
        return(w1);break;        
        case 5:
        return(w1);break;
        case 6:
        return(w2);break;
        case 7:          
        return(w2);break;
        case 8:
        return(w2);break;
        case 9:          
        return(w2);break;        
        case 10:
        return(w);break;
        case 11:
        return(w);break;
        case 12:          
        return(w);break;
        case 13:
        return(w);break;
        case 14:          
        return(w);break;        
        case 15:
        return(w);break;        
        default:return(0);
        }        
    break;
    default:return(0);
    }          
}
size_t FVGaussPoint3D::getNbPoint(size_t order)
{
switch(order)
    {        
    case 0:
    case 1:
    return(1);break;
    case 2:
    case 3:
    return(5);break;
    case 4:
    case 5:
    return(15);break;
    default:return(0);
    }     
}

