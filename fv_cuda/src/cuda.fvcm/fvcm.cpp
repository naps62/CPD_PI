#include <iostream>

#include "FVLib.h"

//	TYPES

typedef
enum {
	UNKNOWN_EXT = 0,
	XML_EXT,
	MSH_EXT
}
extension;

//	PROTOTYPES

void print_usage (string command, ostream& out);
extension parse_extension (string ext_str);
void msh2xml (string filename_in, string filename_out);
void xml2msh (string filename_in, string filename_out);

//	MACRO FUNCTIONS

#define	usage()	\
	print_usage(argv[0], cerr);

//	FUNCTIONS

int main(int argc, char *argv[])
{
	string file_in;		//	input file name
	string file_out;	//	output file name
	extension ext_in;	//	input file extension
	extension ext_out;	//	output file extension

	//	check number of arguments
	if (argc < 3) 
	{
		usage();
		return 1;
	}

	file_in = argv[1];
	file_out = argv[2];

	// parse extensions
	ext_in = parse_extension( file_in.substr( file_in.length() - 4 , 4 ) );
	ext_out = parse_extension( file_out.substr( file_out.length() - 4 , 4 ) );
	if ( !ext_in )
	{
		cerr
			<<	"Input extension unknown"
			<<	endl;
	}
	if ( !ext_out )
	{
		cerr
			<<	"Output extension unknown"
			<<	endl;
	}
	if ( !ext_in || !ext_out )
	{
		usage();
		return 1;
	}

	// convert
	if ( ext_in == MSH_EXT && ext_out == XML_EXT )
		msh2xml(file_in, file_out);
	else if ( ext_in == XML_EXT && ext_out == MSH_EXT )
		xml2msh(file_in, file_out);
	else
	{
		cout
			<<	"No conversion required."
			<<	endl;
	}

	return 0;
}

/*
	Print the command usage
*/
void print_usage (string command, ostream& out)
{
	out
		<<	"Usage:"
		<<	endl
		<<	"\tConvert msh->xml:"
		<<	endl
		<<	"\t\t"
		<<	command
		<<	" <input>.msh <output>.xml"
		<<	endl
		<<	"\tConvert xml->msh:"
		<<	"\t\t"
		<<	command
		<<	" <input>.xml <output>.msh"
		<<	endl;
}

/*
	Known extensions:
	.msh	->	gmsh Mesh file
	.xml	->	XML Mesh file
*/
extension parse_extension (string ext_str)
{
	const char *s;
	
	s = ext_str.c_str();

	switch ( s[0] )
	{
		case '.':
		switch ( s[1] )
		{
			case 'm':
			switch ( s[2])
			{
				case 's':
				switch ( s[3] )
				{
					case 'h':
					switch ( s[4] )
					{
						case '\0':
						return MSH_EXT;

						default:
						return UNKNOWN_EXT;
					}

					default:
					return UNKNOWN_EXT;
				}

				default:
				return UNKNOWN_EXT;
			}

			case 'x':
			switch ( s[2] )
			{
				case 'm':
				switch ( s[3] )
				{
					case 'l':
					switch ( s[4] )
					{
						case '\0':
						return XML_EXT;

						default:
						return UNKNOWN_EXT;
					}

					default:
					return UNKNOWN_EXT;
				}

				default:
				return UNKNOWN_EXT;
			}

			default:
			return UNKNOWN_EXT;
		}

		default:
		return UNKNOWN_EXT;
	}
}

/*
	Convert msh to xml
*/
void msh2xml (string filename_in, string filename_out)
{
	int dim;
	Gmsh mg;

	cout
		<< "Converting msh->xml"
		<< endl;

	mg.readMesh( filename_in.c_str() );
	dim = mg.getDim();
	switch ( dim )
	{
		case 1:
		{
			FVMesh1D m;
			m.Gmsh2FVMesh(mg); 
			m.setName("fvcm convertor");  
			m.write( filename_out.c_str() );
		}
		break;

		case 2:
		{
			FVMesh2D m;
			m.Gmsh2FVMesh(mg); 
			m.setName("fvcm convertor");  
			m.write( filename_out.c_str() );
		}
		break;

		case 3:
		{
			FVMesh3D m;
			m.Gmsh2FVMesh(mg); 
			m.setName("fvcm convertor");  
			m.write( filename_out.c_str() );
		}
		break;

		default:
		cerr
			<< "msh2xml error: Bad mesh dimension ("
			<< dim
			<< ')'
			<< endl;
	}
}

/*
	Convert xml to msh
*/
void xml2msh (string filename_in, string filename_out)
{
	FVMesh1D m1;
	FVMesh2D m2;
	FVMesh3D m3;
	Gmsh mg;

	//if ( m3.read( filename_in.c_str() ) == FVOK )
	//	mg.FVMesh2Gmsh(m3);
	if ( m2.read( filename_in.c_str() ) == FVOK )
		mg.FVMesh2Gmsh(m2);      
	if ( m1.read( filename_in.c_str() ) == FVOK )
		mg.FVMesh2Gmsh(m1);

	mg.writeMesh( filename_out.c_str() );
}

