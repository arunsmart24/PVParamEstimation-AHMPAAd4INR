function [BestX,BestF,HisBestFit,VisitTable]=hybrid_AHA_mpa_three_NR4_noise(MaxIt,nPop,dataset)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FunIndex: The index of function.                    %
% MaxIt: The maximum number of iterations.            %
% nPop: The size of hummingbird population.           %
% PopPos: The position of population.                 %
% PopFit: The fitness of population.                  %
% Dim: The dimensionality of prloblem.                %
% BestX: The best solution found so far.              %
% BestF: The best fitness corresponding to BestX.     %
% HisBestFit: History best fitness over iterations.   %
% Low: The low boundary of search space               %
% Up: The up boundary of search space.                %
% VisitTable: The visit table.                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Dim=9;
        Low=[0 0 1 1 1 0 0 0 0];
        Up=[0.5 100 2 2 2 1 1e-6 1e-6 1e-6];
    PopPos=zeros(nPop,Dim);
    PopFit=zeros(1,nPop);
    prob=zeros(1,nPop);
    minprob=zeros(1,nPop);
    maxprob=zeros(1,nPop);
    stepsize=zeros(nPop,Dim);
    
    It=1;
    
    
    for i=1:nPop
        PopPos(i,:)=rand(1,Dim).*(Up-Low)+Low;
        PopFit(i)=objfuntdnr4_noise(PopPos(i,:), dataset, 1, MaxIt);
    end
    

    BestF=inf;
    BestX=[];

    for i=1:nPop
        if PopFit(i)<=BestF
            BestF=PopFit(i);
            BestX=PopPos(i,:);
        end
    end

    % Initialize visit table
    HisBestFit=zeros(MaxIt,1);
    VisitTable=zeros(nPop) ;
    VisitTable(logical(eye(nPop)))=NaN;    
    
    for It=1:MaxIt
        
        Elite=repmat(BestX,nPop,1);
        theta=2;
        sigma=0.5;
        c1=theta*sin((1-It/MaxIt)*pi/2)+sigma;
        c2=theta*cos((1-It/MaxIt)*pi/2)+sigma;
        DirectVector=zeros(nPop,Dim);% Direction vector/matrix

        for i=1:nPop
            r=rand;
            if r<1/3     % Diagonal flight
                RandDim=randperm(Dim);
                if Dim>=3
                    RandNum=ceil(rand*(Dim-2)+1);
                else
                    RandNum=ceil(rand*(Dim-1)+1);
                end
                DirectVector(i,RandDim(1:RandNum))=1;
            else
                if r>2/3  % Omnidirectional flight
                    DirectVector(i,:)=1;
                else  % Axial flight
                    RandNum=ceil(rand*Dim);
                    DirectVector(i,RandNum)=1;
                end
            end

            if rand<0.5   % Sine Cosine Guided foraging
                [MaxUnvisitedTime,TargetFoodIndex]=max(VisitTable(i,:));
                MUT_Index=find(VisitTable(i,:)==MaxUnvisitedTime);
                if length(MUT_Index)>1
                    [~,Ind]= min(PopFit(MUT_Index));
                    TargetFoodIndex=MUT_Index(Ind);
                end

                newPopPos=c2*rand*PopPos(TargetFoodIndex,:)+c1*rand*(randn*DirectVector(i,:).*...
                    (PopPos(i,:)-PopPos(TargetFoodIndex,:)));
                newPopPos=SpaceBound(newPopPos,Up,Low);
                newPopFit=objfuntdnr4_noise(newPopPos, dataset, It, MaxIt);
                
                
                if newPopFit<PopFit(i)
                    PopFit(i)=newPopFit;
                    PopPos(i,:)=newPopPos;
                    VisitTable(i,:)=VisitTable(i,:)+1;
                    VisitTable(i,TargetFoodIndex)=0;
                    VisitTable(:,i)=max(VisitTable,[],2)+1;
                    VisitTable(i,i)=NaN;
                else
                    VisitTable(i,:)=VisitTable(i,:)+1;
                    VisitTable(i,TargetFoodIndex)=0;
                end
            else    
               
                if It==1 || prob(i)<=rhi(i)% Directional Territorial foraging 
                    b1=randi(nPop);
                    if b1~=i
                        neigh=PopPos(b1,:);
                    elseif b1==nPop
                        neigh=PopPos(b1-1,:);
                    else
                        neigh=PopPos(b1+1,:);
                    end
                    for index=1:Dim
                        if DirectVector(i,index)==0   
                            newPopPos(index)=PopPos(i,index)+...
                                rand*DirectVector(i,index)*PopPos(i,index);
                        elseif DirectVector(i,index)==1 
                            newPopPos(index)=PopPos(i,index)+...
                                rand*DirectVector(i,index)*(PopPos(i,index)-neigh(index));
                        elseif DirectVector(i,index)==-1 
                            newPopPos(index)=PopPos(i,index)-...
                                rand*DirectVector(i,index)*(PopPos(i,index)-neigh(index));
                        end
                    end
                  
                else %(MPA phase)
                    RB=randn(nPop,Dim);
                        for j1=1:size(PopPos,2) 
                            stepsize(i,j1)=RB(i,j1)*(Elite(i,j1)-RB(i,j1)*PopPos(i,j1));                    
                            newPopPos(j1)=PopPos(i,j1)+0.5*rand*stepsize(i,j1); 
                        end
                    
                end
                            
%                 newPopPos= PopPos(i,:)+randn*DirectVector(i,:).*PopPos(i,:);
                newPopPos=SpaceBound(newPopPos,Up,Low);
                newPopFit=objfuntdnr4_noise(newPopPos, dataset, It, MaxIt);
                
                if newPopFit<PopFit(i)
                    PopFit(i)=newPopFit;
                    PopPos(i,:)=newPopPos;
                    VisitTable(i,:)=VisitTable(i,:)+1;
                    VisitTable(:,i)=max(VisitTable,[],2)+1;
                    VisitTable(i,i)=NaN;
                else
                    VisitTable(i,:)=VisitTable(i,:)+1;
                end
            end
        end

        if mod(It,2*nPop)==0 % Opposition based Migration foraging
            [~, MigrationIndex]=max(PopFit);
            PopPos(MigrationIndex,:) =rand(1,Dim).*(Up-Low)+Low;
            oppPopPos=Up+Low-PopPos(MigrationIndex,:);
            PopFit(MigrationIndex)=objfuntdnr4_noise(PopPos(MigrationIndex,:), dataset, It, MaxIt);
            oppPopFit=objfuntdnr4_noise(oppPopPos,dataset,It,MaxIt);
            if oppPopFit<PopFit(MigrationIndex)
                PopPos(MigrationIndex,:)=oppPopPos;
                PopFit(MigrationIndex)=oppPopFit;
            end
            VisitTable(MigrationIndex,:)=VisitTable(MigrationIndex,:)+1;
            VisitTable(:,MigrationIndex)=max(VisitTable,[],2)+1;
            VisitTable(MigrationIndex,MigrationIndex)=NaN;            
        end

        for i=1:nPop
            if PopFit(i)<BestF
                BestF=PopFit(i);
                BestX=PopPos(i,:);
            end
        end

        HisBestFit(It)=BestF;
        
                for ind=1:nPop
                    prob(ind)=PopFit(ind)/sum(PopFit);
                    if It==1
                        minprob(ind)=prob(ind);
                        maxprob(ind)=prob(ind);
                    end
                        
                    if prob(ind)<minprob(ind)
                        minprob(ind)=prob(ind);
                    elseif prob(ind)>maxprob(ind)
                        maxprob(ind)=prob(ind);
                    end
                end
                for ind=1:nPop
                    rhi(ind)=minprob(ind)+rand.*(maxprob(ind)-minprob(ind));
                end
    end


