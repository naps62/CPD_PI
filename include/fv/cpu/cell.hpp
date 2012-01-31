#ifndef ___FV_CPU_CELL_HPP___
#define ___FV_CPU_CELL_HPP___

namespace fv
{
	namespace cpu
	{

		struct Cell
		{
			double velocity[2];
			double polution;
			double area;
			unsigned edgec;
			unsigned *edges;

			Cell();
			Cell(int edgec);
			~Cell();

			void init(int edgec);
		};
	}
}

#endif//___FV_CPU_CELL_HPP___
