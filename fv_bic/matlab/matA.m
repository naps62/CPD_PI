xs = [0, 0,    0,   -1/3, 1/3];
ys = [0, -1/3, 1/3, 0,    0];

A = [ sum(xs.^2),  sum(xs.*ys), sum(xs);
      sum(xs.*ys), sum(ys.^2),  sum(ys);
      sum(xs),     sum(ys),     5];
  
