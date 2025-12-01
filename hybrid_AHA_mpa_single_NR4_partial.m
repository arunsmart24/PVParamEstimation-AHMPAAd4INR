function [BestX, BestF, HisBestFit, VisitTable] = hybrid_AHA_mpa_single_NR4_partial(MaxIt, nPop,dataset)
%==========================================================================
% Hybrid AHA-MPA Algorithm with Single Diode NR4 Objective (Partial Variant)
%
% Combines:
% - Artificial Hummingbird Algorithm (AHA): flight modes, guided foraging
% - Marine Predators Algorithm (MPA): elite-based movement, migration
% - Adaptive control: visit tables and probability-based switching
%
% Inputs:
%   MaxIt  - Maximum number of iterations
%   nPop   - Population size
%
% Outputs:
%   BestX       - Best solution found
%   BestF       - Best fitness value
%   HisBestFit  - History of best fitness values over iterations
%   VisitTable  - Visit table tracking exploration
%==========================================================================

Dim = 5; % [Rs, Rp, Iph, Io1, a1]
Low = [0, 0, 0, 0, 1];
Up  = [0.5, 100, 1, 1e-6, 2];

% --- Initialization ---
PopPos = rand(nPop, Dim) .* (Up - Low) + Low;
PopFit = zeros(1, nPop);
VisitTable = zeros(nPop);
VisitTable(logical(eye(nPop))) = NaN;

% Evaluate initial population
for i = 1:nPop
    PopFit(i) = objfunsdnr4_partial(PopPos(i,:), dataset, 1, MaxIt);
end

% Identify initial best
[BestF, idx] = min(PopFit);
BestX = PopPos(idx, :);
HisBestFit = zeros(MaxIt, 1);

% Probability arrays for adaptive control
prob = zeros(1, nPop);
minprob = zeros(1, nPop);
maxprob = zeros(1, nPop);
stepsize = zeros(nPop, Dim);
rhi = zeros(1, nPop);

% --- Main Loop ---
for It = 1:MaxIt
    Elite = repmat(BestX, nPop, 1);
    theta = 2; sigma = 0.5;
    c1 = theta * sin((1 - It / MaxIt) * pi / 2) + sigma;
    c2 = theta * cos((1 - It / MaxIt) * pi / 2) + sigma;

    for i = 1:nPop
        % Flight direction vector
        DirectVector = zeros(1, Dim);
        r = rand;
        if r < 1/3
            RandDim = randperm(Dim);
            RandNum = ceil(rand * (Dim - 2) + 1);
            DirectVector(RandDim(1:RandNum)) = 1;
        elseif r > 2/3
            DirectVector(:) = 1;
        else
            RandNum = ceil(rand * Dim);
            DirectVector(RandNum) = 1;
        end

        % Guided foraging (AHA)
        if rand < 0.5
            [~, TargetFoodIndex] = max(VisitTable(i,:));
            MUT_Index = find(VisitTable(i,:) == VisitTable(i,TargetFoodIndex));
            if length(MUT_Index) > 1
                [~, Ind] = min(PopFit(MUT_Index));
                TargetFoodIndex = MUT_Index(Ind);
            end

            newPopPos = c2 * rand * PopPos(TargetFoodIndex,:) + ...
                        c1 * rand * (randn * DirectVector .* ...
                        (PopPos(i,:) - PopPos(TargetFoodIndex,:)));
        else
            % Territorial foraging or MPA phase
            if It == 1 || prob(i) <= rhi(i)
                b1 = randi(nPop);
                neigh = PopPos(mod(b1-1, nPop)+1, :);
                newPopPos = PopPos(i,:);
                for index = 1:Dim
                    if DirectVector(index) == 0
                        newPopPos(index) = PopPos(i,index) + rand * PopPos(i,index);
                    elseif DirectVector(index) == 1
                        newPopPos(index) = PopPos(i,index) + rand * (PopPos(i,index) - neigh(index));
                    elseif DirectVector(index) == -1
                        newPopPos(index) = PopPos(i,index) - rand * (PopPos(i,index) - neigh(index));
                    end
                end
            else
                % MPA elite-based movement
                RB = randn(nPop, Dim);
                newPopPos = PopPos(i,:);
                for j1 = 1:Dim
                    stepsize(i,j1) = RB(i,j1) * (Elite(i,j1) - RB(i,j1) * PopPos(i,j1));
                    newPopPos(j1) = PopPos(i,j1) + 0.5 * rand * stepsize(i,j1);
                end
            end
        end

        % Bound and evaluate
        newPopPos = SpaceBound(newPopPos, Up, Low);
        newPopFit = objfunsdnr4_partial(newPopPos, dataset, It, MaxIt);

        % Update if improved
        if newPopFit < PopFit(i)
            PopFit(i) = newPopFit;
            PopPos(i,:) = newPopPos;
            VisitTable(i,:) = VisitTable(i,:) + 1;
            VisitTable(i,TargetFoodIndex) = 0;
            VisitTable(:,i) = max(VisitTable,[],2) + 1;
            VisitTable(i,i) = NaN;
        else
            VisitTable(i,:) = VisitTable(i,:) + 1;
        end
    end

    % Opposition-based migration
    if mod(It, 2*nPop) == 0
        [~, MigrationIndex] = max(PopFit);
        PopPos(MigrationIndex,:) = rand(1, Dim) .* (Up - Low) + Low;
        oppPopPos = Up + Low - PopPos(MigrationIndex,:);
        PopFit(MigrationIndex) = objfunsdnr4_partial(PopPos(MigrationIndex,:), dataset, It, MaxIt);
        oppPopFit = objfunsdnr4_partial(oppPopPos, dataset, It, MaxIt);
        if oppPopFit < PopFit(MigrationIndex)
            PopPos(MigrationIndex,:) = oppPopPos;
            PopFit(MigrationIndex) = oppPopFit;
        end
        VisitTable(MigrationIndex,:) = VisitTable(MigrationIndex,:) + 1;
        VisitTable(:,MigrationIndex) = max(VisitTable,[],2) + 1;
        VisitTable(MigrationIndex,MigrationIndex) = NaN;
    end

    % Update global best
    [BestF, idx] = min(PopFit);
    BestX = PopPos(idx,:);
    HisBestFit(It) = BestF;

    % Update probabilities
    totalFit = sum(PopFit);
    for ind = 1:nPop
        prob(ind) = PopFit(ind) / totalFit;
        if It == 1
            minprob(ind) = prob(ind);
            maxprob(ind) = prob(ind);
        else
            minprob(ind) = min(minprob(ind), prob(ind));
            maxprob(ind) = max(maxprob(ind), prob(ind));
        end
        rhi(ind) = minprob(ind) + rand * (maxprob(ind) - minprob(ind));
    end
end
end
