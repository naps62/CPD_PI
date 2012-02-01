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

		Cell::Cell(int edgec) : edges(NULL)
		{
			this->init(edgec);
		}

		Cell::~Cell()
		{
			delete this->edges;
		}

		//	private methods

		void Cell::init(
			int edgec)
		{
			this->edgec = edgec;
			if (this->edges)
				delete this->edges;
			if (edgec)
				this->edges = new unsigned[edgec];
			else
				this->edges = NULL;
		}
	}
}
