          v_max = numeric_limits<double>::min();
          for (t = 0; t < tc; ++t)
              v_max = ( max_vel_v[t] > v_max )
                      ? max_vel_v[t]
                      : v_max;
          
          dt = 1.0 / abs( v_max );
