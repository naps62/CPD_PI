#ifndef ___FV_CPU_EDGE_HPP___
#define ___FV_CPU_EDGE_HPP___

namespace fv
{
	namespace cpu
	{
		struct Edge
		{
			double flux;
			double length;
			double velocity;
			unsigned left;
			unsigned right;
		};
	}
}

#endif//___FV_CPU_EDGE_HPP___
