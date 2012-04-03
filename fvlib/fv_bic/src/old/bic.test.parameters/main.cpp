// Test of rapidxml library

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "FVL/FVParameters.h"


int main() {
	FVL::FVParameters params("param.xml");

	cout << params.getString("StringVal") << " " << params.getString("StringVal", "opt") << endl;
	cout << params.getDouble("DoubleVal") << endl;
	cout << params.getInteger("IntVal") << endl;
	cout << params.getUnsigned("UnsignedVal") << endl;
	cout << params.getBoolean("BoolVal") << endl;
}
