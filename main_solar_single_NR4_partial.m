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
% The hybrid model is referred to as AHMPAAd4INR and incorporates partial
% shading.
%
%
clc;
clear;
close all;


MaxIteration=500;
PopSize=500;

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
Vp = dataset(:,1);
Ie = dataset(:,2);
N = length(Vp);


disp('Starting Hybrid Optimization...');

tic
[BestX,BestF,HisBestF]=hybrid_AHA_mpa_single_NR4_partial(MaxIteration,PopSize,dataset);
toc
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontWeight', 'bold');



xlabel('Iterations');
ylabel('Fitness');
title(['Double Diode Model']);
hold on
semilogy(HisBestF,'Color','b','LineWidth',4);
title('Convergence curve')
xlabel('Iteration','FontWeight', 'bold','FontSize',14);
ylabel('Best fitness obtained so far','FontWeight', 'bold','FontSize',14);
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
Ie=dataset(:,2);
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

%%%%%%%%%%%%%%%%%%%%%//// PLOTTING I-V CHARACTERISTIC ////%%%%%%%%%%%%%%%%%
%load IV_characteristic_data_experimental
figure
hold on
grid on
plot(Vp,Ip,Vp,Ie,'o','LineWidth',4)   
hold off
title('I-V characteristics under partial shading')
xlabel('Module voltage (v)','FontWeight', 'bold','FontSize',14)
ylabel('Module current(A)','FontWeight', 'bold','FontSize',14)

Pp=Vp.*Ip;                      %Computing the theoretical PV module power
Pe=Vp.*Ie;                      %Computing the experimental PV module power
%%%%%%%%%%%%%%%%%%%%%//// PLOTTING P-V CHARACTERISTIC ////%%%%%%%%%%%%%%%%%
figure
hold on
grid on
plot(Vp,Pp,Vp,Pe,'o','LineWidth',4)   
hold off
title('P-V characteristics under partial shading')
xlabel('Module voltage (v)','FontWeight', 'bold','FontSize',14)
ylabel('Module power(w)','FontWeight', 'bold','FontSize',14)
%%
Iee=(1/length(Vp))*(sum(Ie));
RMSE=sqrt((1/length(Vp))*sum((Ip-Ie).^2))
MBE=(1/length(Vp))*sum((Ip-Ie).^2)
MAE=(1/length(Vp))*sum(abs((Ip-Ie)))
RR=1-((sum((Ie-Ip).^2))/(sum((Ie-Iee).^2)))
TS=sqrt(((length(Vp)-1)*MBE^2)/(RMSE-MBE)^2)
std_err = std(Ip-Ie)







