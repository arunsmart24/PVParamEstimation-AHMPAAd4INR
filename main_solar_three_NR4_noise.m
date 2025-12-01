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
%
% Developed by: Dr. Arunachalam Sundaram
% %==========================================================================
clc;
clear;
close all;


MaxIteration=500;
PopSize=500;
dataset=[...
-0.2057	0.764	7.64E-01	7.64E-01	7.64E-01	7.58E-01	7.36E-01	8.11E-01
-0.1291	0.762	7.62E-01	7.62E-01	7.63E-01	7.67E-01	7.57E-01	8.03E-01
-0.0588	0.7605	7.61E-01	7.61E-01	7.60E-01	7.56E-01	7.28E-01	7.49E-01
0.0057	0.7605	7.61E-01	7.61E-01	7.60E-01	7.58E-01	7.34E-01	8.24E-01
0.0646	0.76	7.60E-01	7.60E-01	7.61E-01	7.65E-01	7.66E-01	8.17E-01
0.1185	0.759	7.59E-01	7.59E-01	7.60E-01	7.52E-01	7.66E-01	6.98E-01
0.1678	0.757	7.57E-01	7.57E-01	7.58E-01	7.59E-01	7.57E-01	7.19E-01
0.2132	0.757	7.57E-01	7.57E-01	7.57E-01	7.50E-01	7.20E-01	8.25E-01
0.2545	0.7555	7.56E-01	7.55E-01	7.55E-01	7.52E-01	7.66E-01	8.25E-01
0.2924	0.754	7.54E-01	7.54E-01	7.54E-01	7.55E-01	7.71E-01	7.99E-01
0.3269	0.7505	7.50E-01	7.51E-01	7.50E-01	7.47E-01	7.55E-01	7.98E-01
0.3585	0.7465	7.47E-01	7.47E-01	7.46E-01	7.44E-01	7.37E-01	7.02E-01
0.3873	0.7385	7.39E-01	7.39E-01	7.39E-01	7.34E-01	7.57E-01	7.25E-01
0.4137	0.728	7.28E-01	7.28E-01	7.28E-01	7.31E-01	7.61E-01	7.59E-01
0.4373	0.7065	7.07E-01	7.07E-01	7.06E-01	7.07E-01	7.13E-01	6.39E-01
0.459	0.6755	6.76E-01	6.76E-01	6.76E-01	6.72E-01	6.94E-01	6.09E-01
0.4784	0.632	6.32E-01	6.32E-01	6.32E-01	6.29E-01	6.39E-01	5.93E-01
0.496	0.573	5.73E-01	5.73E-01	5.73E-01	5.71E-01	5.52E-01	6.08E-01
0.5119	0.499	4.99E-01	4.99E-01	4.99E-01	5.04E-01	4.93E-01	4.52E-01
0.5265	0.413	4.13E-01	4.13E-01	4.13E-01	4.15E-01	4.04E-01	3.93E-01
0.5398	0.3165	3.17E-01	3.16E-01	3.16E-01	3.16E-01	3.31E-01	2.95E-01
0.5521	0.212	2.12E-01	2.12E-01	2.12E-01	2.13E-01	2.08E-01	2.19E-01
0.5633	0.1035	1.04E-01	1.04E-01	1.04E-01	1.03E-01	1.03E-01	1.02E-01
0.5736	-0.01	-1.00E-02	-1.00E-02	-1.00E-02	-1.01E-02	-9.85E-03	-9.27E-03
0.5833	-0.123	-1.23E-01	-1.23E-01	-1.23E-01	-1.22E-01	-1.24E-01	-1.20E-01
0.59	-0.21	-2.10E-01	-2.10E-01	-2.10E-01	-2.12E-01	-2.11E-01	-2.19E-01
];


tic
[BestX,BestF,HisBestF]=hybrid_AHA_mpa_three_NR4_noise(MaxIteration,PopSize,dataset);
toc
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontWeight', 'bold');



xlabel('Iterations');
ylabel('Fitness');
title('Three Diode Model');
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

a1=BestX(3);              %Specified the diode ideality factor value from population
a2=BestX(4);              %Specified the diode ideality factor value from population
a3=BestX(5);              %Specified the diode ideality factor value from population
Rs=BestX(1);             %Specified the PV series resistance value from population
Rp=BestX(2);             %Specified the PV parallel resistance value from population
Iph=BestX(6);            %Specified the photo current value from population
Io1=BestX(7);             %Specified the diode saturation current value from population
Io2=BestX(8);             %Specified the diode saturation current value from population
Io3=BestX(9);             %Specified the diode saturation current value from population


Vp=dataset(:,1);
Ie=dataset(:,7);%change the column number to simulate the effect of noise
G=1000; 
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
% M=1;
F=Iph-Io1.*[exp((Vp+Ie.*Rs)./(a1*VT))-1]-Io2.*[exp((Vp+Ie.*Rs)./(a2*VT))-1]...
    -Io3.*[exp((Vp+Ie.*Rs)./(a3*VT))-1]-((Vp+Ie.*Rs)./Rp)-Ie;
fd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ie.*Rs)./(a1*VT))))-(Io2.*(Rs/a2*VT).*(exp((Vp+Ie.*Rs)./(a2*VT))))...
    -(Io3.*(Rs/a3*VT).*(exp((Vp+Ie.*Rs)./(a3*VT))))-(Rs/Rp)-1;
Ip=Ie-M.*(F./fd); 

FF=Iph-Io1.*[exp((Vp+Ip.*Rs)./(a1*VT))-1]-Io2.*[exp((Vp+Ip.*Rs)./(a2*VT))-1]...
    -Io3.*[exp((Vp+Ip.*Rs)./(a3*VT))-1]-((Vp+Ip.*Rs)./Rp)-Ip;
fdd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ip.*Rs)./(a1*VT))))-(Io2.*(Rs/a2*VT).*(exp((Vp+Ip.*Rs)./(a2*VT))))...
    -(Io3.*(Rs/a3*VT).*(exp((Vp+Ip.*Rs)./(a3*VT))))-(Rs/Rp)-1;
Ip=Ip-M.*(FF./fdd); 

FFF=Iph-Io1.*[exp((Vp+Ip.*Rs)./(a1*VT))-1]-Io2.*[exp((Vp+Ip.*Rs)./(a2*VT))-1]...
    -Io3.*[exp((Vp+Ip.*Rs)./(a3*VT))-1]-((Vp+Ip.*Rs)./Rp)-Ip;
fddd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ip.*Rs)./(a1*VT))))-(Io2.*(Rs/a2*VT).*(exp((Vp+Ip.*Rs)./(a2*VT))))...
    -(Io3.*(Rs/a3*VT).*(exp((Vp+Ip.*Rs)./(a3*VT))))-(Rs/Rp)-1;
Ip=Ip-M.*(FFF./fddd);

FFFF=Iph-Io1.*[exp((Vp+Ip.*Rs)./(a1*VT))-1]-Io2.*[exp((Vp+Ip.*Rs)./(a2*VT))-1]...
    -Io3.*[exp((Vp+Ip.*Rs)./(a3*VT))-1]-((Vp+Ip.*Rs)./Rp)-Ip;
fdddd=-(Io1.*(Rs/a1*VT).*(exp((Vp+Ip.*Rs)./(a1*VT))))-(Io2.*(Rs/a2*VT).*(exp((Vp+Ip.*Rs)./(a2*VT))))...
    -(Io3.*(Rs/a3*VT).*(exp((Vp+Ip.*Rs)./(a3*VT))))-(Rs/Rp)-1;

for i=1:length(Vp)
% Ip(i)=Ie(i)-M*[(2*F(i)^4)/(fddd(i)+fd(i))]^2;
Ip(i)=Ie(i)-M*(F(i)/fd(i))-M*(F(i)^4/fdddd(i));
end

%%%%%%%%%%%%%%%%%%%%%//// PLOTTING I-V CHARACTERISTIC ////%%%%%%%%%%%%%%%%%
%load IV_characteristic_data_experimental
figure(2);
hold on
grid on
plot(Vp,Ip,Vp,Ie,'ys','LineWidth',4)   
hold off
title('I-V characteristics under partial shading')
xlabel('Module voltage (v)','FontWeight', 'bold','FontSize',14)
ylabel('Module current(A)','FontWeight', 'bold','FontSize',14)

Pp=Vp.*Ip;                      %Computing the theoretical PV module power
Pe=Vp.*Ie;                      %Computing the experimental PV module power
%%%%%%%%%%%%%%%%%%%%%//// PLOTTING P-V CHARACTERISTIC ////%%%%%%%%%%%%%%%%%
figure(3);
hold on
grid on
plot(Vp,Pp,Vp,Pe,'ys','LineWidth',4)   
hold off
title('P-V characteristics under partial shading')
xlabel('Module voltage (v)','FontWeight', 'bold','FontSize',14)
ylabel('Module power(w)','FontWeight', 'bold','FontSize',14)


Iee=(1/length(Vp))*(sum(Ie));
RMSE=sqrt((1/length(Vp))*sum((Ip-Ie).^2))
MBE=(1/length(Vp))*sum((Ip-Ie).^2)
RR=1-((sum((Ie-Ip).^2))/(sum((Ie-Iee).^2)))
% TS=sqrt(((length(Vp)-1)*MBE^2)/(RMSE-MBE)^2)




