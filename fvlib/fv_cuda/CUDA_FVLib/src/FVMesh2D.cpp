#include <iostream>
#include <cstdio>
#include <math.h>
#include "FVMesh2D.h"
#include "Gmsh.h"
FVMesh2D::FVMesh2D()
{
	_nb_vertex=0;_nb_cell=0;_nb_edge=0;_dim=0;
}  


FVMesh2D::FVMesh2D(const char *filename)
{
	FVMesh2D::read(filename);
}


size_t FVMesh2D::read(const char *filename)
{
	_spxml.readXML(filename,_xml); // read the file and put in the XML string
	string key, value,element;
	StringMap attribute;  
	//StringMap::iterator iter;
	size_t code;
	// open  balise FVLIB 
	if (_spxml.openBalise("FVLIB")!=OkOpenBalise)
	{cout<<" No open VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;return(NoOpenBalise);}    
	//  find the FIELD  balise
	code=_spxml.openBalise("MESH");
	if (code!=OkOpenBalise)
	{cout<<" No open MESH balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);} 
	attribute=_spxml.getAttribute();           
	// read the dim of the mesh  
	key=string("dim");
	if(key.empty()) 
	{cout<<" No dim attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
	value= attribute[key];    
	_dim= (unsigned) atoi( value.c_str()); 
	if(_dim!=2)
	{
#ifdef _DEBUGS
		cout<<" dimension do not match:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;
#endif 
		return(FVWRONGDIM);
	} 
	// read the mesh       
	key=string("name");
	if(key.empty()) 
	{cout<<" No name attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
	_name= attribute[key];
	// ------------------------ we read the vertices  ------------------
	code=_spxml.openBalise("VERTEX");    
	if (code!=OkOpenBalise)
	{cout<<" No open VERTEX balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
	attribute=_spxml.getAttribute();     
	// read the number of vertices
	key=string("nbvertex");
	if(key.empty()) 
	{cout<<" No nbvertex attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
	value= attribute[key];
	_nb_vertex=(unsigned) atoi( value.c_str()); 
	// read the Vertices data
	_vertex.resize(_nb_vertex);
	_spxml.data();
	{
		size_t beginDATA=_spxml.getPosition();
		size_t lengthDATA=_spxml.getLength();

		char *thedata, *ptr;
		thedata = new char[ lengthDATA + 1 ];

		//char  thedata[lengthDATA+1],*ptr;
		_xml.copy(thedata,lengthDATA,beginDATA);
		thedata[lengthDATA]=0;
		//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
		ptr=thedata;
		size_t count=0;
		while(count<_nb_vertex)// read the data and put it in the valarray
		{
			_vertex[count].label= strtod(ptr, &ptr);
			_vertex[count].code= strtod(ptr, &ptr);
			_vertex[count].coord.x= strtod(ptr, &ptr);
			_vertex[count].coord.y= strtod(ptr, &ptr);   
			count++; 
		}

		delete thedata;
	}
	// close  Balise   VERTEX
	code=_spxml.closeBalise("VERTEX");
	if (code!=OkCloseBalise)
	{cout<<" No close VERTEX balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}   

	//----------------  we read the edges --------------------------        

	code=_spxml.openBalise("EDGE");    
	if (code!=OkOpenBalise)
	{cout<<" No open EDGE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
	attribute=_spxml.getAttribute();     
	// read the number of cell
	key=string("nbedge");
	if(key.empty()) 
	{cout<<" No nbedge attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
	value= attribute[key];
	_nb_edge=(unsigned) atoi( value.c_str());  

	// read the EDGE data
	_edge.resize(_nb_edge);
	_spxml.data();

	{
		size_t beginDATA=_spxml.getPosition();
		size_t lengthDATA=_spxml.getLength();

		char *thedata, *ptr;
		thedata = new char[ lengthDATA + 1 ];

		//char  thedata[lengthDATA+1],*ptr;
		_xml.copy(thedata,lengthDATA,beginDATA);
		thedata[lengthDATA]=0;
		//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
		ptr=thedata;
		size_t count=0,_label;

		while(count<_nb_edge)// read the data and put it in the valarray
		{
			_label= strtod(ptr, &ptr);
			_edge[_label-1].label=_label;
			_edge[_label-1].code= strtod(ptr, &ptr); 
			_edge[_label-1].nb_vertex= strtod(ptr, &ptr);   // should be always 2
			_edge[_label-1].firstVertex= &(_vertex[strtod(ptr, &ptr)-1]); 
			_edge[_label-1].secondVertex=&(_vertex[strtod(ptr, &ptr)-1]);     
			count++; 
		}

		delete thedata;
	}    
	// close  Balise   EDGE
	code=_spxml.closeBalise("EDGE");
	if (code!=OkCloseBalise)
	{cout<<" No close EDGE balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}         

	//----------------  we read the cells --------------------------

	code=_spxml.openBalise("CELL");    
	if (code!=OkOpenBalise)
	{cout<<" No open CELL balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoOpenBalise);}    
	attribute=_spxml.getAttribute();     
	// read the number of cell
	key=string("nbcell");
	if(key.empty()) 
	{cout<<" No nbcell attribute found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoAttribute);}     
	value= attribute[key];
	_nb_cell=(unsigned) atoi( value.c_str());  

	// read the CELL data
	_cell.resize(_nb_cell);
	_spxml.data();

	{
		size_t beginDATA=_spxml.getPosition();
		size_t lengthDATA=_spxml.getLength();
		
		char *thedata, *ptr;
		thedata = new char[ lengthDATA + 1 ];

		//char  thedata[lengthDATA+1],*ptr;
		_xml.copy(thedata,lengthDATA,beginDATA);
		thedata[lengthDATA]=0;
		//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
		ptr=thedata;
		size_t count=0,_label;

		while(count<_nb_cell)// read the data and put it in the valarray
		{
			_label= strtod(ptr, &ptr);
			_cell[_label-1].label=_label;
			_cell[_label-1].code= strtod(ptr, &ptr);
			_cell[_label-1].nb_edge= strtod(ptr, &ptr);  
			for(size_t i=0;i<_cell[_label-1].nb_edge;i++)
				_cell[_label-1].edge[i]= &(_edge[strtod(ptr, &ptr)-1]);        
			count++; 
		}

		delete thedata;
	}    
	// close  Balise   CELL
	code=_spxml.closeBalise("CELL");
	if (code!=OkCloseBalise)
	{cout<<" No close CELL balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}          
	//---------------------------------------------                
	// close  Balise   MESH
	code=_spxml.closeBalise("MESH");
	if (code!=OkCloseBalise)
	{cout<<" No close MESH balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);} 
	// close  Balise   FVLIB
	code=_spxml.closeBalise("FVLIB");
	if (code!=OkCloseBalise)
	{cout<<" No close FVLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(NoCloseBalise);}        

	//------- we have store all the data, now we complete the vertices and cells information    
	FVMesh2D::complete_data();
	return(FVOK);     
}

void FVMesh2D::complete_data()
{
	// initialize the vertices and compute the centroid and length
	//cout<<"entering in complete_data"<<endl;fflush(NULL);
	//cout<<"step1"<<endl;fflush(NULL);
	for(size_t i=0;i<_nb_edge;i++)
	{
		_edge[i].leftCell=NULL;_edge[i].rightCell=NULL;
		_edge[i].nb_cell=2;
		_edge[i].centroid=(_edge[i].firstVertex->coord+_edge[i].secondVertex->coord)*0.5;
		FVPoint2D<double> u;
		u=_edge[i].firstVertex->coord-_edge[i].secondVertex->coord;
		_edge[i].length=Norm(u);
	}
	//cout<<"step2"<<endl;fflush(NULL);
	// contruct the list of cell for each vertex    
	for(size_t i=0;i<_nb_vertex;i++)
	{
		_vertex[i].nb_cell=0;
	}

	//cout<<"step3"<<endl;fflush(NULL);
	for(size_t i=0;i<_nb_cell;i++)
	{
		_cell[i].perimeter=0.;
		_cell[i].area=0.;
		_cell[i].centroid=0.;
		// conpute centroid  and perimeter for the cell 
		// determine the left and right cell for the vertices
		//cout<<"step3.1 :cell "<<i+1<<" with "<<_cell[i].nb_edge<<" edges"<<endl;fflush(NULL);
		for(size_t j=0;j<_cell[i].nb_edge;j++)
		{ 
			_cell[i].perimeter+=_cell[i].edge[j]->length; 
			_cell[i].centroid+=_cell[i].edge[j]->centroid*_cell[i].edge[j]->length; 
			if(!(_edge[_cell[i].edge[j]->label-1].leftCell)  )  
				_edge[_cell[i].edge[j]->label-1].leftCell=&(_cell[i]);
			else
				_edge[_cell[i].edge[j]->label-1].rightCell=&(_cell[i]);  
		}  
		// compute the centroid of the cell
		_cell[i].centroid/=_cell[i].perimeter;  
		// compute the cell2edge vector and the area
		//cout<<"step3.2"<<endl;fflush(NULL);
		for(size_t j=0;j<_cell[i].nb_edge;j++)
		{
			_cell[i].cell2edge[j]= _cell[i].edge[j]->centroid-_cell[i].centroid;      
			FVPoint2D<double> u,v;  
			u=_cell[i].edge[j]->firstVertex->coord-_cell[i].centroid; 
			v=_cell[i].edge[j]->secondVertex->coord-_cell[i].centroid;
			_cell[i].area+=fabs(Det(u,v))*0.5;
		}
		// update the list of the vertices->cell pointer   
		//cout<<"step3.3"<<endl;fflush(NULL);
		pos_v=0;
		for(size_t j=0;j<_cell[i].nb_edge;j++)
		{
			bool _still_exist;   
			_still_exist=false; 
			for(size_t k=0;k<pos_v;k++)
				if(_cell[i].edge[j]->firstVertex==_cell[i].vertex[k])  _still_exist=true;
			if(!_still_exist) {_cell[i].vertex[pos_v]=_cell[i].edge[j]->firstVertex;pos_v++;}
			_still_exist=false;  
			for(size_t k=0;k<pos_v;k++)
				if(_cell[i].edge[j]->secondVertex==_cell[i].vertex[k])  _still_exist=true;
			if(!_still_exist) {_cell[i].vertex[pos_v]=_cell[i].edge[j]->secondVertex;pos_v++;}    
		}
		_cell[i].nb_vertex=pos_v;   
		//cout<<"step3.4"<<endl;fflush(NULL);
		for(size_t j=0;j<_cell[i].nb_vertex;j++)
		{
			size_t pos;
			pos=_cell[i].vertex[j]->label-1;
			_vertex[pos].cell[_vertex[pos].nb_cell]=&(_cell[i]); 
			_vertex[pos].nb_cell++;
		}  
	}
	//cout<<"step4"<<endl;fflush(NULL);    
	//  we compute the normal from left to rigth  
	_nb_boundary_edge=0;
	for(size_t i=0;i<_nb_edge;i++) 
	{
		//cout<<"edge number "<<i<<endl;   
		FVPoint2D<double> u,v;  
		double no;
		u=_edge[i].firstVertex->coord-_edge[i].secondVertex->coord;
		no=Norm(u);
		_edge[i].normal.x=u.y;_edge[i].normal.y=-u.x;
		_edge[i].normal/=no;
		v=_edge[i].centroid-_edge[i].leftCell->centroid;
		if(_edge[i].normal*v<0) _edge[i].normal*=-1.; 
		if(! (_edge[i].rightCell)) {_boundary_edge.push_back(&(_edge[i]));_nb_boundary_edge++;} 
	}
	// cout<<"end of complete_data"<<endl;fflush(NULL);    
}


size_t FVMesh2D::write(const char *filename)
{
	if((_nb_cell==0) || (_nb_edge==0) || (_nb_vertex==0))
	{
		cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
		cout<<" there is no mesh to save"<<endl;
		return(FVERROR);
	}    
	ofstream  out_file;    
	out_file.open(filename);
	if(!out_file.is_open())
	{
		cout<<" error in file:"<<filename<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; 
		cout<<" can not create the file"<<endl;
		return(FVNOFILE);
	}

	out_file<<"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"<<endl;
	out_file<<"<FVLIB>"<<endl; 
	out_file<<"    <MESH dim=\"2\"    name=\""<<_name<<"\">"<<endl;
	out_file<<"         <VERTEX nbvertex=\""<<_nb_vertex<<"\">"<<endl;
	for(size_t i=0;i<_nb_vertex;i++)
	{
		out_file<< setw(FVCHAMPINT)<<_vertex[i].label<< setw(FVCHAMPINT)<< _vertex[i].code;  
		out_file<<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << _vertex[i].coord.x;
		out_file<<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << _vertex[i].coord.y<<endl; 
	}
	out_file<<"         </VERTEX>"<<endl; 
	out_file<<"         <EDGE nbedge=\""<<_nb_edge<<"\">"<<endl;
	for(size_t i=0;i<_nb_edge;i++)
	{
		out_file<< setw(FVCHAMPINT)<<_edge[i].label<< setw(FVCHAMPINT)<< _edge[i].code
			<< setw(FVCHAMPINT)<< _edge[i].nb_vertex; 
		out_file<<setw(FVCHAMPINT) << _edge[i].firstVertex->label<<setw(FVCHAMPINT) << _edge[i].secondVertex->label<<endl; 
	} 
	out_file<<"         </EDGE>"<<endl; 
	out_file<<"         <CELL nbcell=\""<<_nb_cell<<"\">"<<endl;
	for(size_t i=0;i<_nb_cell;i++)
	{
		out_file<< setw(FVCHAMPINT)<<_cell[i].label<< setw(FVCHAMPINT)<< _cell[i].code
			<< setw(FVCHAMPINT)<< _cell[i].nb_edge;
		for(size_t j=0;j<_cell[i].nb_edge;j++)        
			out_file<<setw(FVCHAMPINT) << _cell[i].edge[j]->label<<setw(FVCHAMPINT); 
		out_file<<endl; 
	} 
	out_file<<"         </CELL>"<<endl;      
	out_file<<"    </MESH>"<<endl;
	out_file<<"</FVLIB>"<<endl;          
	out_file.close(); 
	return(FVOK);
}




void FVMesh2D::Gmsh2FVMesh( Gmsh &m) // convert a Gmsh struct into a FVMesh2D
{
	if (m.getDim()!=2)
	{
		cout<<" Dimension don't match. The Gmsh mesh should contain only 0D or 1D elements"<<endl;
		return;
	}  
	_nb_vertex=m.getNbNode();
	_name="mesh2D  from gmsh file"; 
	_vertex.resize(_nb_vertex);
	for(size_t i=0;i<_nb_vertex;i++)
	{
		_vertex[i].label=m.getNode(i)->label;
		_vertex[i].coord.x=m.getNode(i)->coord.x;
		_vertex[i].coord.y=m.getNode(i)->coord.y;
		_vertex[i].code=0;  // default value
	}   
	_nb_cell=0;_nb_edge=0;   
	_cell.resize(0);_edge.resize(0);
	_cell.reserve(m.getNbElement());_edge.reserve(m.getNbElement()*3);
	FVCell2D cell;FVEdge2D edge;
	size_t pos_dim2=0;
	bool exist;
	//cout<<"step1"<<endl;

	for(size_t i=0;i<m.getNbElement();i++)
	{  
		GMElement *ptr_el=m.getElement(i);  
		if  (ptr_el->type_element==15) 
		{
			pos_dim2++;
			ptr_el->label=_vertex[ptr_el->node[0]-1].label; // label is no longer GMElement label but  give 
			// the label of the vertex in the _vertex table
		} 
		if  (ptr_el->type_element==1)
		{ 
			pos_dim2++;
			edge.firstVertex=&(_vertex[ptr_el->node[0]-1]);
			edge.secondVertex=&(_vertex[ptr_el->node[1]-1]);
			edge.nb_vertex=2;
			exist=false;
			for(size_t j=0;j<_edge.size();j++)// look if the edge still exist
				if(isEqual(&_edge[j],&edge)) {exist=true;ptr_el->label=j+1;}
			if(!exist)
			{
				ptr_el->label=edge.label=_edge.size()+1;    // label is no longer GMElement label but  give
				// the label of the edge in the _edge table
				edge.code=0;  // we see the code later
				_edge.push_back(edge); // it is a real new edge so save it
			}   
		}
		if  (ptr_el->type_element==2 ||ptr_el->type_element==3 )  // treat   a triangle or a quadrangle
		{
			//cout<<"creating cell"<<endl;fflush(NULL);           
			for(size_t j=0;j<=ptr_el->type_element;j++)
			{
				FVEdge2D edge;
				edge.firstVertex=&(_vertex[ptr_el->node[j]-1]);
				edge.secondVertex=&(_vertex[ptr_el->node[(j+1)%(ptr_el->type_element+1)]-1]);
				edge.nb_vertex=2;
				exist=false;
				for(size_t k=0;k<_edge.size();k++)// look if the edge still exist
					if(isEqual(&_edge[k],&edge))  
					{
						cell.edge[j]=&_edge[k];exist=true;                    
					} // if exist we keep the pointer
				if(!exist) // if not exist create it
				{  
					edge.label=_edge.size()+1;             
					edge.code=0;  // we see the code later
					_edge.push_back(edge); // it is a real new edge so save it
					cell.edge[j]=&_edge[_edge.size()-1]; // and keep the pointer
				}
			}
			// at that stage we have all the edge now chesk the code
			cell.nb_edge=ptr_el->type_element+1;  
			exist=false;
			for(size_t j=0;j<_cell.size();j++)// look if the cell still exist
			{    
				if(isEqual(&_cell[j],&cell))                 
				{
					exist=true;
					if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
					{_cell[j].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}
					if (ptr_el->code_physical/MINUS_ONE_DIM==1)
					{_cell[j].setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}  
					if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
						_cell[j].code=ptr_el->code_physical;
				}
			}      
			if(!exist) 
			{
				//cout<<"the cell (gmsh label)"<<ptr_el->label<< " does not exist"<<endl;  fflush(NULL);
				//cout<<"the edge are "<<cell.edge[0]->label<<" "<<cell.edge[1]->label<<" "<<cell.edge[2]->label<<" "<<endl;

				ptr_el->label=cell.label=_cell.size()+1;    // label is no longer GMElement label but  give
				// the label of the cell in the _cell table   

				if (ptr_el->code_physical/MINUS_ONE_DIM==2)               
				{cell.setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}
				if (ptr_el->code_physical/MINUS_ONE_DIM==1)
				{cell.setCode2Edge(ptr_el->code_physical%MINUS_ONE_DIM);}   
				if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
					cell.code=ptr_el->code_physical;
				_cell.push_back(cell); // it is a real new edge so save it

			}  
		}      
	}   
	_nb_edge=_edge.size();    
	_nb_cell=_cell.size();  
	//cout<<"step2"<<endl;  
	// all the edge and cell are created    
	// now we treat the code cascade backward







	for(size_t cont=pos_dim2;cont>0;cont--) // we skip of 1 becaus NEGATIVE NUMBER DOES NOT ESXIST with size_t
	{  
		size_t i=cont-1; // astuce to allow deacresing counter   
		GMElement *ptr_el=m.getElement(i); 

		// we treat the edge

		if(ptr_el->type_element==1)
		{ 
			if (ptr_el->code_physical/MINUS_ONE_DIM==1)
			{_edge[ptr_el->label-1].setCode2Vertex(ptr_el->code_physical%MINUS_ONE_DIM);}  // we set the Vertex code 
			if (ptr_el->code_physical/MINUS_ONE_DIM==0)                      
				_edge[ptr_el->label-1].code=ptr_el->code_physical;   // we set the edge code 
		}       
		//we treat the node       
		if(ptr_el->type_element==15) 
		{
			_vertex[ptr_el->label-1].code=ptr_el->code_physical;  // we set the Vertex code 
		}

	} 








	//cout<<"step3"<<endl;        
	// complete the data
	/*
	   cout<<"I have made "<< _nb_edge<<" and "<<_nb_cell<<" cells"<<endl;

	   for(size_t i=0;i<_nb_edge;i++)
	   {
	   cout<<" edge no "<< _edge[i].label<<" with "<<_edge[i].nb_vertex<<" vertices "
	   <<_edge[i].firstVertex->label<<" and "<<_edge[i].secondVertex->label<<endl,
	   cout<<"the code is "<< _edge[i].code<<endl;
	   }   
	 */
	/*
	   cout<<"======================"<<endl;
	   for(size_t i=0;i<_nb_cell;i++)
	   {
	   cout<<" cell no "<< _cell[i].label<<" with "<<_cell[i].nb_edge<<" edges "<<", the code is "<< _cell[i].code<<endl;
	   cout<<" list od the edges"<<endl;
	   for(size_t j=0;j<_cell[i].nb_edge;j++)
	   cout<<" edge no "<<_cell[i].edge[j]->label<<", ";
	   cout<<endl;
	   } 
	 */
	FVMesh2D::complete_data();     
}
