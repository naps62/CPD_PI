#ifndef _H_PARAMETERS
#define _H_PARAMETERS

#include <string>
using namespace std;

/*
Parameters: holds the data from the parameter file
*/
typedef
struct _parameters
{
	struct
	{
		string mesh;
		string velocity;
		struct
		{
			string initial;
			string output;
		} polution;
	} filenames;
	struct
	{
		double final;
	} time;
	struct
	{
		int jump;
	} iterations;
	struct
	{
		double threshold;
	} computation;
}
Parameters;

//	END TYPES

//	BEGIN FUNCTIONS

/*
   Reads the parameters file.
   */
Parameters read_parameters (string parameter_filename);

#endif // _H_PARAMETERS
