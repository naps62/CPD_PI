#include <iostream>
#include "FVLib.h"

int main(int argc, char *argv[])
{
if (argc < 4) 
        {
        cout<<"Wrong format"<<endl;    
        cout<<"PLEASE use Convert data gmsh->xml: fvcd mesh.xml data.xml data.msh (-c/v) "<<endl;
        exit(0);
        }        
FVVect<double> x=0,y=0,z=0; 
string    file_mesh=argv[1];
string    file_in=argv[2];
string    file_out=argv[3];    
string    type_name,name;
double    time=0;
size_t nbv=0,type=0;
if(argc==5) type_name=argv[4]; else type_name="-c"; // default data on cell
if(type_name.compare("-v")) type=CELL; else type=VERTEX;
//cout<<file_mesh<<" "<<file_in<<" "<<file_out<<" "<<dim<<" "<<type<<endl;
size_t l;
l=file_mesh.length();
string label_mesh;label_mesh.insert(0,file_mesh,l-4,4);
l=file_in.length();
string label_in;label_in.insert(0,file_in,l-4,4);
l=file_out.length();
string label_out;label_out.insert(0,file_out,l-4,4);
if(label_mesh.compare(".xml"))  {cout<<"Wrong Mesh format, .xml label required"<<endl; exit(0);}
if(label_in.compare(".xml"))  {cout<<"Wrong input file format, .xml label required"<<endl; exit(0);}
if(label_out.compare(".msh")) {cout<<"Wrong output file format, .msh label required"<<endl; exit(0);} 
//
FVMesh1D m1;
FVMesh2D m2;
FVMesh3D m3;
Gmsh mg;
//cout<<"reading the xml mesh file"<<endl;fflush(NULL);
if(m3.read(file_mesh.c_str())==FVOK) mg.FVMesh2Gmsh(m3);
if(m2.read(file_mesh.c_str())==FVOK) mg.FVMesh2Gmsh(m2);      
if(m1.read(file_mesh.c_str())==FVOK) mg.FVMesh2Gmsh(m1);
//cout<<"read ok, now we write the msh mesh"<<endl;
mg.writeMesh(file_out.c_str());
FVio data_in(file_in.c_str(),FVREAD);
// convert the data
size_t result=FVOK;
while (result==FVOK)
    {
     result=data_in.get(x, y, z, time, name);
     nbv=data_in.getNbVect();
     if(result==FVOK) 
         {
         switch(nbv)
              {
              case 0:
              cout<<"  nbvec not found in the file"<<endl;    
              result=FVERROR;
              break;    
              case 1:
              mg.writeVector(x,type,name.c_str(),time);    
              break;
              case 2:
              mg.writeVector(x,y,type,name.c_str(),time);              
              break;
              case 3:
              mg.writeVector(x,y,z,type,name.c_str(),time); 
              break;
              }
          }    
     }  
mg.close();   
}

