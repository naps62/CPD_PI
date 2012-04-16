i = 1;
j = 2;
ij = 3;

%a = 1;
%b = 2;
%c = 3;

%ABC = [-1 0 0.2]
u = [1 0 1];

u_min = min(u(i), u(j));
u_max = max(u(i), u(j));

if ((u(ij) - u(i)) / (u(j) - u(i)) < 0)
   psi = 0;
else
    den = u(j) - u(i);
    num = u(ij) - u(i);
    
    psi = min(1, den/num);
end

%out = ['u_ij* = ', num2str(u(ij)), ', psi = ', num2str(psi)];

disp(psi);