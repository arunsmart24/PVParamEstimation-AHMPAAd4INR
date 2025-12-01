function Fit=objfunsdnr4irr(Xnew,C_Iter)

% objfunsdnr4 - Fitness evaluation using 4th-order Newton-Raphson method
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


% Experimental dataset [Voltage, Current]
dataset =[...
-0.2057	0.1528	0.3056	0.4584	0.6112	0.764
-0.1291	0.1524	0.3048	0.4572	0.6096	0.762
-0.0588	0.1521	0.3042	0.4563	0.6084	0.7605
0.0057	0.1521	0.3042	0.4563	0.6084	0.7605
0.0646	0.152	0.304	0.456	0.608	0.76
0.1185	0.1518	0.3036	0.4554	0.6072	0.759
0.1678	0.1514	0.3028	0.4542	0.6056	0.757
0.2132	0.1514	0.3028	0.4542	0.6056	0.757
0.2545	0.1511	0.3022	0.4533	0.6044	0.7555
0.2924	0.1508	0.3016	0.4524	0.6032	0.754
0.3269	0.1501	0.3002	0.4503	0.6004	0.7505
0.3585	0.1493	0.2986	0.4479	0.5972	0.7465
0.3873	0.1477	0.2954	0.4431	0.5908	0.7385
0.4137	0.1456	0.2912	0.4368	0.5824	0.728
0.4373	0.1413	0.2826	0.4239	0.5652	0.7065
0.459	0.1351	0.2702	0.4053	0.5404	0.6755
0.4784	0.1264	0.2528	0.3792	0.5056	0.632
0.496	0.1146	0.2292	0.3438	0.4584	0.573
0.5119	0.0998	0.1996	0.2994	0.3992	0.499
0.5265	0.0826	0.1652	0.2478	0.3304	0.413
0.5398	0.0633	0.1266	0.1899	0.2532	0.3165
0.5521	0.0424	0.0848	0.1272	0.1696	0.212
0.5633	0.0207	0.0414	0.0621	0.0828	0.1035
0.5736	-0.002	-0.004	-0.006	-0.008	-0.01
0.5833	-0.0246	-0.0492	-0.0738	-0.0984	-0.123
0.59	-0.042	-0.084	-0.126	-0.168	-0.21
];


% --- 1. Constants and Parameter Extraction ---
Tc = 306.15; % Cell Temperature in Kelvin
q = 1.60217646e-19; % Electron charge (C)
k = 1.3806503e-23;  % Boltzmann constant (J/K)
VT = (k * Tc) / q;  % Thermal voltage

% Extract Parameters
Rs  = Xnew(1);  % Series resistance
Rp  = Xnew(2);  % Parallel resistance
Iph = Xnew(3);  % Photocurrent
Io1 = Xnew(4);  % Diode saturation current
a1  = Xnew(5);  % Ideality factor
kVT = a1 * VT;  % Combined term (n*VT)

Vp = dataset(:,1);
Ie = dataset(:,3);%Change the column corresponding to irradiance level
N = length(Vp);

% --- 2. Adaptive Damping Factors (For Novelty/Search Control) ---
% These factors decay from 1 (exploration) toward 0 (exploitation)
M1 = 1 - (C_Iter / MaxIteration)^2; % Damping for 1st-order NR term
M2 = 1 - (C_Iter / MaxIteration)^2; % Damping for 4th-order correction term
rho = 100; 

% --- 3. Iterative Solver Parameters ---
Tolerance = 1e-9;         % Convergence criterion
MaxLocalIterations = 20; 
Ip = zeros(N,1);          % Array to store calculated currents
dEdI = Rs / kVT;          % Rs / (n * VT) - used frequently in derivatives

% --- 4. Adaptive Fourth-Order Newton-Raphson Solver Loop ---
for j = 1:N 
    V = Vp(j);
    I_calc = Ie(j); % Initial guess (measured current)
    
    for iter = 1:MaxLocalIterations 
        exp_term = exp((V + I_calc * Rs) / kVT);
        
        % 4.1. Implicit Function F(I)
        F_I = Iph - Io1 * (exp_term - 1) - ((V + I_calc * Rs) / Rp) - I_calc;
        
        % 4.2. First Derivative F'(I)
        FD_I = -(Io1 * dEdI) * exp_term - (Rs / Rp) - 1;
        
        % 4.3. Fourth Derivative F''''(I)
        % F''''(I) = -Io1 * (dEdI)^4 * exp(term)
        FDDDD_I = -(Io1 * dEdI^4) * exp_term; 
        
        % 4.4. Numerical Stability Checks (CRITICAL for high-order methods)
        
        % Check FD_I (for standard NR term)
        if abs(FD_I) < 1e-12 
            FD_I = sign(FD_I) * 1e-12; 
        end
        
        % Check FDDDD_I (for 4th-order correction term)
        % This check prevents division by near-zero, the likely cause of poor results
        if abs(FDDDD_I) < 1e-12 
             FDDDD_I = sign(FDDDD_I) * 1e-12; 
        end

        % 4.5. Adaptive Fourth-Order Step Calculation (The Novelty)
        % Delta_I = -M1* (F/F') - M2 * rho * (F^4 / F'''')
        Delta_I = -M1 * (F_I / FD_I) - M2 * (rho * F_I^4 / FDDDD_I);

        % Overshoot Limit: Cap the maximum step size to prevent divergence
        MaxStep = 0.1; 
        Delta_I = max(min(Delta_I, MaxStep), -MaxStep); 
        
        I_calc_new = I_calc + Delta_I;

        % Check for convergence
        if abs(Delta_I) < Tolerance
            break; 
        end
        I_calc = I_calc_new; % Update current for next iteration
    end
    Ip(j) = I_calc;
end

% --- 5. Compute Fitness (RMSE) and Penalties ---
Fit = sqrt(mean((Ie - Ip).^2));

% Penalize invalid solutions (crucial for optimization stability)
% Check for NaN or Inf (which signals solver crash) and assign a very high penalty.
if isnan(Fit) || isinf(Fit) || Fit < 1e-12
    Fit = 1e6; % Assign a very high penalty
end

end
