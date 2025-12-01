function Fit=objfunddnr4(Xnew,dataset,C_Iter, MaxIteration)

%==========================================================================
% Adaptive Fourth-Order Newton-Raphson (Ad4INR) Objective Function for DDM
%
% This function calculates the RMSE fitness score for the Double Diode Model
% (DDM) using a customized 4th-order iterative solver.
% DDM has 7 parameters: [Iph, Io1, Io2, Rs, Rp, a1, a2]
%==========================================================================

% --- 1. Constants and Parameter Extraction (7 Parameters) ---
Tc = 306.15; % Cell Temperature in Kelvin
q = 1.60217646e-19; % Electron charge (C)
k = 1.3806503e-23;  % Boltzmann constant (J/K)
VT = (k * Tc) / q;  % Thermal voltage

% Extract Parameters using the specified mapping:
Iph = Xnew(1);  % Photocurrent
Io1 = Xnew(2);  % Diode 1 saturation current
Io2 = Xnew(3);  % Diode 2 saturation current
Rs  = Xnew(4);  % Series resistance
Rp  = Xnew(5);  % Parallel resistance
a1  = Xnew(6);  % Diode 1 ideality factor
a2  = Xnew(7);  % Diode 2 ideality factor

% Combined thermal voltage terms
kVT1 = a1 * VT; 
kVT2 = a2 * VT;

Vp = dataset(:,1);
Ie = dataset(:,2);
N = length(Vp);

% --- 2. Adaptive Damping Factors ---
% These factors decay from 1 (exploration) toward 0 (exploitation)
M1 = 1 - (C_Iter / MaxIteration)^2; % Damping for 1st-order NR term
M2 = 1 - (C_Iter / MaxIteration)^2; % Damping for 4th-order correction term
rho = 100; % Scaling factor for the 4th-order term

% --- 3. Iterative Solver Parameters ---
Tolerance = 1e-9;         % Convergence criterion
MaxLocalIterations = 20; 
Ip = zeros(N,1);          % Array to store calculated currents

% Frequent derivative terms
dEdI1 = Rs / kVT1; % Rs / (a1 * VT)
dEdI2 = Rs / kVT2; % Rs / (a2 * VT)

% --- 4. Adaptive Fourth-Order Newton-Raphson Solver Loop ---
for j = 1:N 
    V = Vp(j);
    I_calc = Ie(j); % Initial guess (measured current)
    
    for iter = 1:MaxLocalIterations 
        
        % Exponentials for both diodes
        exp_term1 = exp((V + I_calc * Rs) / kVT1);
        exp_term2 = exp((V + I_calc * Rs) / kVT2);
        
        % 4.1. Implicit Function F(I) for DDM
        % F(I) = Iph - I_D1 - I_D2 - I_Rp - I = 0
        F_I = Iph - Io1 * (exp_term1 - 1) - Io2 * (exp_term2 - 1) - ((V + I_calc * Rs) / Rp) - I_calc;
        
        % 4.2. First Derivative F'(I)
        % F'(I) = dF/dI
        FD_I = -(Io1 * dEdI1) * exp_term1 - (Io2 * dEdI2) * exp_term2 - (Rs / Rp) - 1;
        
        % 4.3. Fourth Derivative F''''(I)
        % F''''(I) = -(Io1 * dEdI1^4) * exp_term1 - (Io2 * dEdI2^4) * exp_term2
        FDDDD_I = -(Io1 * dEdI1^4) * exp_term1 - (Io2 * dEdI2^4) * exp_term2; 
        
        % 4.4. Numerical Stability Checks (CRITICAL)
        
        % Check FD_I 
        if abs(FD_I) < 1e-12 
            FD_I = sign(FD_I) * 1e-12; 
        end
        
        % Check FDDDD_I
        if abs(FDDDD_I) < 1e-12 
             FDDDD_I = sign(FDDDD_I) * 1e-12; 
        end

        % 4.5. Adaptive Fourth-Order Step Calculation
        % Delta_I = -M1* (F/F') - M2 * rho * (F^4 / F'''')
        Delta_I = -M1 * (F_I / FD_I) - M2 * (rho * F_I^4 / FDDDD_I);

        % Overshoot Limit
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

% Penalize invalid solutions
if isnan(Fit) || isinf(Fit) || Fit < 1e-12
    Fit = 1e6; % Assign a very high penalty
end
end