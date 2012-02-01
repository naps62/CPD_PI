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
			unsigned edge_count;
			unsigned *edges;

			Cell();
			Cell(int edge_count);
			~Cell();

			void init(int edge_count);
		};
	}
}

#endif//___FV_CPU_CELL_HPP___
