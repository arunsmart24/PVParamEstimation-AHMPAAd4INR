function Fit=objfunsdnr4_partial(Xnew, dataset, C_Iter, MaxIteration)
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
Ie = dataset(:,2);
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
