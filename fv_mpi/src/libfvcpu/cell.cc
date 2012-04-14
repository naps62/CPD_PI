#include <cstdlib>

#include <fv/cpu/cell.hpp>

namespace fv
{
	namespace cpu
	{
		class Edge;

		Cell::Cell() : edges(NULL)
		{
			this->init(0);
		}

		Cell::Cell(int edge_count) : edges(NULL)
		{
			this->init(edge_count);
		}

		Cell::~Cell()
		{
			delete this->edges;
		}

		//	private methods

		void Cell::init(
			int edge_count)
		{
			this->edge_count = edge_count;
			if (this->edges)
				delete[] this->edges;
			if (edge_count)
				this->edges = new unsigned[ edge_count ];
			else
				this->edges = NULL;
		}
	}
}
