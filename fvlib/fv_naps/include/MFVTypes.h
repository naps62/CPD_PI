#ifndef _M_FV_TYPES
#define _M_FV_TYPES

/**
 * FV_DualPtr holds two pointers
 * cpu_ptr to handle data in cpu
 * gpu_ptr, when not null, holds corresponding ptr to the gpu copied data
 *
 * TODO: handle gpu_ptr destruction
 */
class FV_DualPtr {
	private:
		friend class FV_GPU_Point2D;

	public:
		double *cpu_ptr;
		double *gpu_ptr;

		// basic constructors/destructors
		FV_DualPtr()						 	{ zero_ptrs(); }
		FV_DualPtr(double *cpu, double *gpu)	{ cpu_ptr = cpu; gpu_ptr = gpu; }
		FV_DualPtr(FV_DualPtr &p)				{ cpu_ptr = p.cpu_ptr; gpu_ptr = p.gpu_ptr; }

		~FV_DualPtr() { delete_cpu(); }

		/**
		 * allocs cpu_ptr with a double[size] array
		 */
		void alloc_cpu(unsigned int size) {
			if (cpu_ptr != NULL)
				delete_cpu();
			cpu_ptr = new double[size];
		}

		/**
		 * deallocates cpu_ptr memory
		 */
		void delete_cpu() {
			if (cpu_ptr != NULL) {
				delete cpu_ptr;
				cpu_ptr = NULL;
			}
		}

	private:
		/**
		 * Sets both pointers to NULL
		 */
		void zero_ptrs() {
			cpu_ptr = NULL;
			gpu_ptr = NULL;
		}

};


/**
 * FV_GPU_Point2D holds two FV_DualPtr, one for x coord and one for y
 */
class FV_GPU_Point2D {
	public:
		FV_DualPtr x;
		FV_DualPtr y;

		// basic constructors/destructors
		FV_GPU_Point2D()							 	{ zero_ptrs(); }
		FV_GPU_Point2D(FV_DualPtr p1, FV_DualPtr p2)	{ x = FV_DualPtr(p1); y = FV_DualPtr(p2); }
		FV_GPU_Point2D(FV_GPU_Point2D &p)		{ x = p.x; y = p.y; }


		/**
		 * allocs cpu_ptr with a double[size] array
		 */
		void alloc_cpu(unsigned int size) {
			x.alloc_cpu(size);
			y.alloc_cpu(size);
		}

		/**
		 * deallocates cpu_ptr memory
		 */
		void delete_cpu() {
			x.delete_cpu();
			y.delete_cpu();
		}

	private:
		/**
		 * Sets both pointers to NULL
		 */
		void zero_ptrs() {
			x.zero_ptrs();
			y.zero_ptrs();
		}
};

#endif // _M_FV_TYPES
