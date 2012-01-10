// ------ PFVio.cpp ------
// S. CLAIN 2011/07
#include <FVio.h>


/*---------- Default constructor -------*/
FVio::FVio() 
{
	_is_open=false;
	_field_name="";
	_nbvec=0;
}


/*---------- constructor -------*/
FVio::FVio(const char *filename, int access)
{
	FVio::open( filename, access); 
	_nbvec=0;
}

/* ---------- destructor -------*/
FVio::~FVio() // le destructeur ferme le fichier
{
	if (_is_open) FVio::close();    
}

/*--------- open the file ---------*/
void FVio::open(const char *filename, int access )
{
	_is_open=true;
	_access=access;   
	if (_access == FVWRITE) // we want to write
	{    
		_of.open(filename);
		_of<<"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"<<endl;
		_of<<"<FVLIB>"<<endl;     
		return;
	}    
	if (_access == FVREAD)  // we want to read
	{ 
		_spxml.readXML(filename,_xml); // read the file and put in the XML string
		// open  balise FVLIB 
		if ((_spxml.openBalise("FVLIB")!=OkOpenBalise) )
		{cout<<" No open VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;}
		return;
	}
	cout<<"access code "<<_access<<" unknown"<<endl;       
}


/*---------- close the file ---------*/
void FVio::close()
{
	if (!_is_open) 
	{
		cout<<"file is not open, can not close"<<endl;
		return;
	}
	_is_open=false;
	if (_access == FVWRITE) // we write
	{    
		_of<<"</FVLIB>"<<endl;          
		_of.close(); 
		return; 
	}    
	if (_access == FVREAD)  // we  read
	{ 
		// close  Balise   FVLIB   
		//  if ((_spxml.closeBalise("FVLIB")!=OkCloseBalise) && (!_of.eof()))
		//   {cout<<" No close VFLIB balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl;}
		_of.close();           
		return;
	}
	cout<<"access code "<<_access<<" unknown"<<endl;
}


/* -------  write 1 vector -------*/
void FVio::put(FVVect <double> &u, const double time, const string &name)
{
	if(_access!=FVWRITE)
	{
		cout<<" wrong access code: should be FVWRITE"<<endl; return;
	}
	_of<<"    <FIELD ";
	_of<<"size=\""<<u.size()<<"\"   ";
	_of<<"nbvec=\""<<1<<"\"   ";    
	_of<<"time=\""<<time<<"\"   ";
	_of<<"name=\""<<name<<"\">"<<endl;
	for(size_t i=0;i<u.size(); i++)
		_of <<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << u[i] ;

	_of<<endl;    
	_of<<"    </FIELD>"<<endl;;    
}
/* -------  write 2 vectors -------*/
void FVio::put(FVVect <double> &u,FVVect <double> &v, const double time, const string &name)
{
	if(_access!=FVWRITE)
	{
		cout<<" wrong access code: should be FVWRITE"<<endl; return;
	}
	if(u.size()!=v.size())
	{
		cout<<" vectors must have same size "<<endl; return;
	}     
	_of<<"    <FIELD ";
	_of<<"size=\""<<u.size()<<"\"   ";
	_of<<"nbvec=\""<<2<<"\"   ";    
	_of<<"time=\""<<time<<"\"   ";
	_of<<"name=\""<<name<<"\">"<<endl;
	for(size_t i=0;i<v.size(); i++)
		_of <<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << u[i] <<" "<<v[i] ;

	_of<<endl;    
	_of<<"    </FIELD>"<<endl;;    
}
// the FVPoint2D version 
void FVio::put(FVVect <FVPoint2D<double> >&u, const double time,const  string &name)
{
	if(_access!=FVWRITE)
	{
		cout<<" wrong access code: should be FVWRITE"<<endl; return;
	}   
	_of<<"    <FIELD ";
	_of<<"size=\""<<u.size()<<"\"   ";
	_of<<"nbvec=\""<<2<<"\"   ";    
	_of<<"time=\""<<time<<"\"   ";
	_of<<"name=\""<<name<<"\">"<<endl;
	for(size_t i=0;i<u.size(); i++)
		_of <<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << u[i].x <<" "<<u[i].y ;

	_of<<endl;    
	_of<<"    </FIELD>"<<endl;;    
}
/* -------  write 3 vectors -------*/
void FVio::put(FVVect <double> &u,FVVect <double> &v,FVVect <double> &w, const double time, const string &name)
{
	if(_access!=FVWRITE)
	{
		cout<<" wrong access code: should be FVWRITE"<<endl; return;
	}
	if(u.size()!=v.size() ||u.size()!=w.size() )
	{
		cout<<" vectors must have same size "<<endl; return;
	}     
	_of<<"    <FIELD ";
	_of<<"size=\""<<u.size()<<"\"   ";
	_of<<"nbvec=\""<<3<<"\"   ";    
	_of<<"time=\""<<time<<"\"   ";
	_of<<"name=\""<<name<<"\">"<<endl;
	for(size_t i=0;i<v.size(); i++)
		_of <<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << u[i] <<" "<<v[i] <<" "<<w[i];

	_of<<endl;    
	_of<<"    </FIELD>"<<endl;;    
}

// the FVPoint3D version 
void FVio::put(FVVect <FVPoint3D<double> >&u, const double time, const string &name)
{
	if(_access!=FVWRITE)
	{
		cout<<" wrong access code: should be FVWRITE"<<endl; return;
	}   
	_of<<"    <FIELD ";
	_of<<"size=\""<<u.size()<<"\"   ";
	_of<<"nbvec=\""<<3<<"\"   ";    
	_of<<"time=\""<<time<<"\"   ";
	_of<<"name=\""<<name<<"\">"<<endl;
	for(size_t i=0;i<u.size(); i++)
		_of <<scientific << setprecision(FVPRECISION) << setw(FVCHAMP) << u[i].x <<" "<<u[i].y<<" "<<u[i].z ;

	_of<<endl;    
	_of<<"    </FIELD>"<<endl;;    
}



/* -------  read 1 vector -------*/
size_t FVio::get(FVVect <double> &u,  double &time, string &name)
{
	if(_access!=FVREAD)
	{
		cout<<" wrong access code: should be FVREAD"<<endl; return(FVERROR);
	}    
	if(!_is_open)
	{
		cout<<" file is not open can not read"<<endl; return(FVNOFILE);
	}         

	//FVio::showXML();
	string key, value,element;
	StringMap attribute;  
	StringMap::iterator iter;
	size_t code;    

	//  find the FIELD  balise
	code=_spxml.openBalise("FIELD");
	if (code!=OkOpenBalise)
	{return(FVENDFILE);}
	//  read the attributes
	attribute=_spxml.getAttribute();
	// read the size of vect
	key=string("size");
	if(key.empty()) 
	{cout<<" No size balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}     
	value= attribute[key];
	_sizevec=(unsigned) atoi( value.c_str()); 
	// read the number  of vect
	key=string("nbvec");
	if(key.empty()) {_nbvec=1;}
	else 
	{
		value= attribute[key];    
		_nbvec=(unsigned) atoi( value.c_str()); 
	}
	// read the time 
	key=string("time");
	if(key.empty()) {time=_time=0.;}
	else
	{
		value= attribute[key];
		time=_time=(unsigned) atof( value.c_str());
	}
	// read the name 
	key=string("name");
	name=_name= attribute[key];
	// we have all the attribute, now read the vector
	if (u.size()<_sizevec) u.resize(_sizevec);    
	_spxml.data();
	size_t beginDATA=_spxml.getPosition();
	size_t lengthDATA=_spxml.getLength();

	char *thedata, *ptr;
	thedata = new char[lengthDATA + 1];

	//char  thedata[lengthDATA+1],*ptr;
	_xml.copy(thedata,lengthDATA,beginDATA);
	thedata[lengthDATA]=0;
	//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
	ptr=thedata;
	size_t count=0;
	double ignore_val;ignore_val=0;ignore_val++;
	while(count<_sizevec)// read the data and put it in the valarray
	{
		u[count]= strtod(ptr, &ptr);
		if(_nbvec>=2) ignore_val=strtod(ptr, &ptr); // skip the second double
		if(_nbvec>=3) ignore_val=strtod(ptr, &ptr); // skip the third double
		count++; 
	}
	// close  Balise   FIELD
	code=_spxml.closeBalise("FIELD");
	if (code!=OkCloseBalise)
	{cout<<" No close FIELD balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}     

	delete thedata;

	return(FVOK);    
}


/* -------  read 2 vectors -------*/    
size_t FVio::get(FVVect <double> &u, FVVect <double> &v, double &time, string &name)
{
	if(_access!=FVREAD)
	{
		cout<<" wrong access code: should be FVREAD"<<endl; return(FVERROR);
	} 
	if(!_is_open)
	{
		cout<<" file is not open can not read"<<endl; return(FVNOFILE);
	}       
	string key, value,element;
	StringMap attribute;  
	StringMap::iterator iter;
	size_t code;    

	//  find the FIELD  balise
	code=_spxml.openBalise("FIELD");
	if (code!=OkOpenBalise)
	{return(FVENDFILE);}
	//  read the attributes
	attribute=_spxml.getAttribute();
	// read the size of vect
	key=string("size");
	if(key.empty()) 
	{cout<<" No size balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}     
	value= attribute[key];
	_sizevec=(unsigned) atoi( value.c_str()); 
	// read the number  of vect
	key=string("nbvec");
	if(key.empty()) {_nbvec=1;}
	else 
	{
		value= attribute[key];    
		_nbvec=(unsigned) atoi( value.c_str()); 
	}
	// read the time 
	key=string("time");
	if(key.empty()) {time=_time=0.;}
	else
	{
		value= attribute[key];
		time=_time=(unsigned) atof( value.c_str());
	}
	// read the name 
	key=string("name");
	name=_name= attribute[key];
	// we have all the attribute, now read the vector
	if (u.size()<_sizevec) u.resize(_sizevec);  
	if (v.size()<_sizevec) v.resize(_sizevec);  
	_spxml.data();
	size_t beginDATA=_spxml.getPosition();
	size_t lengthDATA=_spxml.getLength();

	char *thedata, *ptr;
	thedata = new char[lengthDATA + 1];

	//char  thedata[lengthDATA+1],*ptr;
	_xml.copy(thedata,lengthDATA,beginDATA);
	thedata[lengthDATA]=0;
	//for(size_t i=0;i<lengthDATA;i++) cout<<thedata[i]; cout<<endl;
	ptr=thedata;
	size_t count=0;
	double ignore_val;ignore_val=0;ignore_val++;
	while(count<_sizevec)// read the data and put it in the valarray
	{
		u[count]= strtod(ptr, &ptr);
		if(_nbvec>=2) v[count]=strtod(ptr, &ptr); // take the second double
		if(_nbvec>=3) ignore_val=strtod(ptr, &ptr); // skip the third double
		//cout<<u[count]<<" and "<<v[count];
		count++; 
	}
	// close  Balise   FIELD
	code=_spxml.closeBalise("FIELD");
	if (code!=OkCloseBalise)
	{cout<<" No close FIELD balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}         
	delete thedata;
	return(FVOK);
}    


//FVPoint2D version
size_t FVio::get(FVVect <FVPoint2D<double> >&u, double &time, string &name)
{
	if(_access!=FVREAD)
	{
		cout<<" wrong access code: should be FVREAD"<<endl; return(FVERROR);
	} 
	if(!_is_open)
	{
		cout<<" file is not open can not read"<<endl; return(FVNOFILE);
	}       
	string key, value,element;
	StringMap attribute;  
	StringMap::iterator iter;
	size_t code;    

	//  find the FIELD  balise
	code=_spxml.openBalise("FIELD");
	if (code!=OkOpenBalise)
	{return(FVENDFILE);}
	//  read the attributes
	attribute=_spxml.getAttribute();
	// read the size of vect
	key=string("size");
	if(key.empty()) 
	{cout<<" No size balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}     
	value= attribute[key];
	_sizevec=(unsigned) atoi( value.c_str()); 
	// read the number  of vect
	key=string("nbvec");
	if(key.empty()) {_nbvec=1;}
	else 
	{
		value= attribute[key];    
		_nbvec=(unsigned) atoi( value.c_str()); 
	}
	// read the time 
	key=string("time");
	if(key.empty()) {time=_time=0.;}
	else
	{
		value= attribute[key];
		time=_time=(unsigned) atof( value.c_str());
	}
	// read the name 
	key=string("name");
	name=_name= attribute[key];
	// we have all the attribute, now read the vector
	if (u.size()<_sizevec) u.resize(_sizevec);  
	_spxml.data();
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
	double ignore_val;ignore_val=0;ignore_val++;
	while(count<_sizevec)// read the data and put it in the valarray
	{
		u[count].x= strtod(ptr, &ptr);
		if(_nbvec>=2) u[count].y=strtod(ptr, &ptr); // take the second double
		if(_nbvec>=3) ignore_val=strtod(ptr, &ptr); // skip the third double
		//cout<<u[count]<<" and "<<v[count];
		count++; 
	}
	// close  Balise   FIELD
	code=_spxml.closeBalise("FIELD");
	if (code!=OkCloseBalise)
	{cout<<" No close FIELD balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}         

	delete thedata;

	return(FVOK);
} 
/* -------  read 3 vectors -------*/     
size_t FVio::get(FVVect <double> &u, FVVect <double> &v, FVVect <double> &w, double &time, string &name)
{
	if(_access!=FVREAD)
	{
		cout<<" wrong access code: should be FVREAD"<<endl; return(FVERROR);
	}  
	if(!_is_open)
	{
		cout<<" file is not open can not read"<<endl; return(FVNOFILE);
	}       

	string key, value,element;
	StringMap attribute;  
	StringMap::iterator iter;
	size_t code;    

	//  find the FIELD  balise
	code=_spxml.openBalise("FIELD");
	if (code!=OkOpenBalise)
	{return(FVENDFILE);}
	//  read the attributes
	attribute=_spxml.getAttribute();
	// read the size of vect
	key=string("size");
	if(key.empty()) 
	{cout<<" No size balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}     
	value= attribute[key];
	_sizevec=(unsigned) atoi( value.c_str()); 
	// read the number  of vect
	key=string("nbvec");
	if(key.empty()) {_nbvec=1;}
	else 
	{
		value= attribute[key];    
		_nbvec=(unsigned) atoi( value.c_str()); 
	}
	// read the time 
	key=string("time");
	if(key.empty()) {time=_time=0.;}
	else
	{
		value= attribute[key];
		time=_time=(unsigned) atof( value.c_str());
	}
	// read the name 
	key=string("name");
	name=_name= attribute[key];
	// we have all the attribute, now read the vector
	if (u.size()<_sizevec) u.resize(_sizevec);  
	if (v.size()<_sizevec) v.resize(_sizevec); 
	if (w.size()<_sizevec) w.resize(_sizevec); 
	_spxml.data();
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
	while(count<_sizevec)// read the data and put it in the valarray
	{
		u[count]= strtod(ptr, &ptr);
		if(_nbvec>=2) v[count]=strtod(ptr, &ptr); // take the second double
		if(_nbvec>=3) w[count]=strtod(ptr, &ptr); // take the third double
		//cout<<u[count]<<" and "<<v[count]<<" et "<<w[count];
		count++; 
	}
	// close  Balise   FIELD
	code=_spxml.closeBalise("FIELD");
	if (code!=OkCloseBalise)
	{cout<<" No close FIELD balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}         

	delete thedata;

	return(FVOK);
}

// the FVPoint3D version
size_t FVio::get(FVVect <FVPoint3D<double> >&u, double &time, string &name)
{
	if(_access!=FVREAD)
	{
		cout<<" wrong access code: should be FVREAD"<<endl; return(FVERROR);
	}  
	if(!_is_open)
	{
		cout<<" file is not open can not read"<<endl; return(FVNOFILE);
	}       

	string key, value,element;
	StringMap attribute;  
	StringMap::iterator iter;
	size_t code;    

	//  find the FIELD  balise
	code=_spxml.openBalise("FIELD");
	if (code!=OkOpenBalise)
	{return(FVENDFILE);}
	//  read the attributes
	attribute=_spxml.getAttribute();
	// read the size of vect
	key=string("size");
	if(key.empty()) 
	{cout<<" No size balise found :"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}     
	value= attribute[key];
	_sizevec=(unsigned) atoi( value.c_str()); 
	// read the number  of vect
	key=string("nbvec");
	if(key.empty()) {_nbvec=1;}
	else 
	{
		value= attribute[key];    
		_nbvec=(unsigned) atoi( value.c_str()); 
	}
	// read the time 
	key=string("time");
	if(key.empty()) {time=_time=0.;}
	else
	{
		value= attribute[key];
		time=_time=(unsigned) atof( value.c_str());
	}
	// read the name 
	key=string("name");
	name=_name= attribute[key];
	// we have all the attribute, now read the vector
	if (u.size()<_sizevec) u.resize(_sizevec);  
	_spxml.data();
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
	while(count<_sizevec)// read the data and put it in the valarray
	{
		u[count].x= strtod(ptr, &ptr);
		if(_nbvec>=2) u[count].y=strtod(ptr, &ptr); // take the second double
		if(_nbvec>=3) u[count].z=strtod(ptr, &ptr); // take the third double
		//cout<<u[count]<<" and "<<v[count]<<" et "<<w[count];
		count++; 
	}
	// close  Balise   FIELD
	code=_spxml.closeBalise("FIELD");
	if (code!=OkCloseBalise)
	{cout<<" No close FIELD balise found:"<<"  in "<<__FILE__<< " line "<<__LINE__<<endl; return(FVERROR);}

	delete thedata;

	return(FVOK);       
}

