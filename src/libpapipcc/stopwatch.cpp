#include <papi/stopwatch.hpp>
#include <papi.h>

#define NOW() PAPI_get_real_nsec()

namespace papi
{
	namespace time
	{
		namespace real
		{
			Stopwatch::Stopwatch()
			: _running(false)
			, _last(0)
			{
				_total = 0;
				_overhead = 0;
				this->start();
				this->stop();
				_overhead = _total;
				_total = 0;
			}



			void
			Stopwatch::start()
			{
				if ( ! _running )
				{
					_last = 0;
					_running = true;
					_begin = NOW();
				}
			}



			void
			Stopwatch::stop()
			{
				_end = NOW();
				if ( _running )
				{
					_running = false;
					_last = _end - _begin - _overhead;
					_total += _last;
				}
			}



			void
			Stopwatch::reset()
			{
				_last = 0;
				if ( ! _running )
				{
					_total = 0;
				}
			}



			long long int
			Stopwatch::finish()
			{
				this->stop();
				long long int timens = this->last();
				this->reset();
				return timens;
			}



			void
			Stopwatch::toggle()
			{
				if ( _running )
				{
					this->stop();
				}
				else
				{
					this->start();
				}
			}



			long long int
			Stopwatch::total()
			const
			{
				return _total;
			}



			long long int
			Stopwatch::last()
			const
			{
				return _last;
			}



		}// end of namespace 'real'
	}// end of namespace 'time'
}// end of namespace 'papi'
