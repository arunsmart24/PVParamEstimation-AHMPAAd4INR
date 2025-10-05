function Fit=objfunsdnr4_partial(Xnew,C_Iter)
% Fitness evaluation using 4th-order Newton-Raphson method
%
% Inputs:
%   Xnew    - Candidate solution vector [Rs, Rp, Iph, Io1, a1]
%   C_Iter  - Current iteration number
%
% Output:
%   Fit     - Root Mean Square Error (RMSE) between estimated and measured current
%
% This function estimates the output current of a PV cell using the
% Single Diode Model and refines it using a fourth-order Newton-Raphson method.
%==========================================================================

global dataset MaxIteration

% Constants
Tc = 3.0615e+02;
q = 1.60217646e-19;
k = 1.3806503e-23;
VT=(k*Tc)/q;
% Extract parameters from candidate solution

a1=Xnew(5);              %Diode ideality factor 
Rs=Xnew(1);             %PV series resistance 
Rp=Xnew(2);             %PV parallel resistance 
Iph=Xnew(3);            % Photo current 
Io1=Xnew(4);             %Diode saturation current
 
% Experimental dataset [Voltage, Current]


dataset=[...
-0.2057	0.6922
-0.1291	0.7020
-0.0588	0.7405
0.0057	0.7020
0.0646	0.6456
0.1185	0.5947
0.1678	0.4756
0.2132	0.3872
0.2545	0.2345
0.2924	0.3024
0.3269	0.3505
0.3585	0.4165
0.3873	0.4485
0.4137	0.4902
0.4373	0.4975
0.459	0.4864
0.4784	0.4832
0.496	0.4662
0.5119	0.447
0.5265	0.413
0.5398	0.3165
0.5521	0.212
0.5633	0.1035
0.5736	-0.01
0.5833	-0.123
0.59	-0.21
];


Vp = dataset(:,1);            % Voltage points
Ie = dataset(:,2);            % Measured current points
N = length(Vp);               % Number of data points

% Adaptive damping factor
M = exp(-(5 * C_Iter / MaxIteration)^2.5);

% First NR iteration
F  = Iph - Io1 .* (exp((Vp + Ie .* Rs) ./ (a1 * VT)) - 1) - ((Vp + Ie .* Rs) ./ Rp) - Ie;
fd = -Io1 .* (Rs / (a1 * VT)) .* exp((Vp + Ie .* Rs) ./ (a1 * VT)) - (Rs / Rp) - 1;
Ip = Ie - M .* (F ./ fd);

% Second NR iteration
FF  = Iph - Io1 .* (exp((Vp + Ip .* Rs) ./ (a1 * VT)) - 1) - ((Vp + Ip .* Rs) ./ Rp) - Ip;
fdd = -Io1 .* (Rs / (a1 * VT)) .* exp((Vp + Ip .* Rs) ./ (a1 * VT)) - (Rs / Rp) - 1;
Ip  = Ip - M .* (FF ./ fdd);

% Third NR iteration
FFF  = Iph - Io1 .* (exp((Vp + Ip .* Rs) ./ (a1 * VT)) - 1) - ((Vp + Ip .* Rs) ./ Rp) - Ip;
fddd = -Io1 .* (Rs / (a1 * VT)) .* exp((Vp + Ip .* Rs) ./ (a1 * VT)) - (Rs / Rp) - 1;
Ip   = Ip - M .* (FFF ./ fddd);

% Fourth NR iteration
FFFF  = Iph - Io1 .* (exp((Vp + Ip .* Rs) ./ (a1 * VT)) - 1) - ((Vp + Ip .* Rs) ./ Rp) - Ip;
fdddd = -Io1 .* (Rs / (a1 * VT)) .* exp((Vp + Ip .* Rs) ./ (a1 * VT)) - (Rs / Rp) - 1;

% Final refinement using 4th-order correction
for i = 1:N
    Ip(i) = Ie(i) - M * (F(i) / fd(i)) - M * (F(i)^4 / fdddd(i));
end

% Compute RMSE as fitness value
Fit = sqrt((1 / N) * sum((Ie - Ip).^2));

end
