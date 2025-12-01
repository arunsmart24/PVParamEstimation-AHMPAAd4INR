%==========================================================================
% AHMPAAd4INR Hybrid Algorithm for PV Model Parameter Estimation
%
% This MATLAB program implements a hybrid optimization approach combining:
% - Marine Predators Algorithm (MPA) [Faramarzi et al., 2020]
% - Artificial Hummingbird Algorithm (AHA) [Zhao et al., 2021]
% - Adaptive Fourth-Order Newton-Raphson Method (Ad4INR)
%
% Original MPA Source:
% A. Faramarzi, M. Heidarinejad, S. Mirjalili, A.H. Gandomi,
% "Marine Predators Algorithm: A Nature-inspired Metaheuristic",
% Expert Systems with Applications, DOI: 10.1016/j.eswa.2020.113377
%
% Original AHA Source:
% W. Zhao, L. Wang, S. Mirjalili,
% "Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications",
% Computer Methods in Applied Mechanics and Engineering (2021), DOI: 10.1016/j.cma.2021.114194
%
% This work hybridizes MPA and AHA with an adaptive fourth-order Newton-Raphson refinement
% to estimate parameters in Single Diode (SD), Double Diode (DD), and Three Diode (TD) PV models.
% The hybrid model is referred to as AHMPAAd4INR.


clc;
clear;
close all;
global dataset MaxIteration

MaxIteration=500;
PopSize=500;

tic
[BestX,BestF,HisBestF]=hybrid_AHA_mpa_single_NR4_Irr(MaxIteration,PopSize);
toc


xlabel('Iterations');
ylabel('Fitness');
title(['Double Diode Model']);
hold on
semilogy(HisBestF,'Color','b','LineWidth',4);
title('Convergence curve')
xlabel('Iteration');
ylabel('Best fitness obtained so far');
axis tight
grid off
box on
legend('Hybrid AHA and MPA')

display(['The best location of Hybrid AHA and MPA is: ', num2str(BestX)]);
display(['The best fitness of Hybrid AHA and MPA is: ', num2str(BestX)]);
format shortE

a1=BestX(5);              %Specified the diode ideality factor value from population
Rs=BestX(1);             %Specified the PV series resistance value from population
Rp=BestX(2);             %Specified the PV parallel resistance value from population
Iph=BestX(3);            %Specified the photo current value from population
Io1=BestX(4);             %Specified the diode saturation current value from population


Vp=dataset(:,1);
Ie=dataset(:,3); %Change the column corresponding to irradiance level
G=400; %change this value to 200 (second column) 400 (3rd) 600(4th) 800 (5th) and 1000 (6th Column)
T=306.15;
radiation=[G];                                                         %///&&&Array of solar radiation in (w/m^2)///&&&%
cell_temperature=[T];
solar_radiation=radiation./1000;
% Nsc=36;                             %Number of cells are connected in series per module
k=1.3806503*10^-23;                 %Boltzmann constant (J/K)
q=1.60217646*10^-19;                %Electron charge in (Columb)
G=solar_radiation;                 %Reading the solar radiation values one by one
Tc=cell_temperature; 
VT=(k*Tc)/q;        %Diode thermal voltage (v)
%%%%%%%%//// Computing the theoretical current using NR method ////%%%%%%%%
Ip=zeros(size(Vp));
N = length(Vp); % define Rp, Rs, Vth, a – Ideality factor, Io, Ipv, …
M=exp(-(5*MaxIteration/MaxIteration)^2.5);
% M1=1-(MaxIteration/(MaxIteration))^1;
% M2=1-(MaxIteration/(MaxIteration))^1;
F=Iph-Io1.*[exp((Vp+Ie.*Rs)./(a1*VT))-1]-((Vp+Ie.*Rs)./Rp)-Ie;
fd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ie.*Rs)./(a1*VT))))-(Rs/Rp)-1;
Ip=Ie-M.*(F./fd); 

FF=Iph-Io1.*[exp((Vp+Ip.*Rs)./(a1*VT))-1]-((Vp+Ip.*Rs)./Rp)-Ip;
fdd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ip.*Rs)./(a1*VT))))-(Rs/Rp)-1;
Ip=Ip-M.*(FF./fdd); 

FFF=Iph-Io1.*[exp((Vp+Ip.*Rs)./(a1*VT))-1]-((Vp+Ip.*Rs)./Rp)-Ip;
fddd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ip.*Rs)./(a1*VT))))-(Rs/Rp)-1;
Ip=Ip-M.*(FFF./fddd);

FFFF=Iph-Io1.*[exp((Vp+Ip.*Rs)./(a1*VT))-1]-((Vp+Ip.*Rs)./Rp)-Ip;
fdddd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ip.*Rs)./(a1*VT))))-(Rs/Rp)-1;

for i=1:length(Vp)
% Ip(i)=Ie(i)-M*[(2*F(i)^4)/(fddd(i)+fd(i))]^2;
Ip(i)=Ie(i)-M*(F(i)/fd(i))-M*(F(i)^4/fdddd(i));
end


% Compute predicted and experimental power
Pp = Vp .* Ip;   % Predicted power
Pe = Vp .* Ie;   % Experimental powe


% Compute metrics
Iee = mean(Ie);
RMSE = sqrt(mean((Ip - Ie).^2));
MBE  = mean(Ip - Ie);
MAE  = mean(abs(Ip - Ie));
RR   = 1 - (sum((Ie - Ip).^2) / sum((Ie - Iee).^2));
TS   = sqrt(((length(Vp) - 1) * MBE^2) / (RMSE - MBE)^2);
std_err = std(Ip - Ie);



