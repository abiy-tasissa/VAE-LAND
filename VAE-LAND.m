% -------------------------------------------------------------------------
% This script runs LAND on a latent feature extracted from a VAE
% (Input is a .csv file of the latent data obtained from VAE)
% See also the associated Python file
% The LAND code, necessary for this script, can be found here
% https://jmurphy.math.tufts.edu/Code/LAND_Public_V1.0.zip
% -------------------------------------------------------------------------
%% Compare LAND, and DH for different datasets and budgets.  
% All datasets are taken from the paper :
% Maggioni, M. and J.M. Murphy
% "Learning by Active Nonlinear Diffusion." 
% arXiv preprint arXiv:1905.12989 (2019).

clear all;
close all;
% Specify location of LAND_Public folder
addpath(genpath('C:/home/abiyo\Desktop\LAND_Public\LAND_Public'))

%% Select dataset

%{
DataSet='SyntheticBottleneck';
M=0;
N=0;
Budget=1:1:170;
PeakOpts.DiffusionOpts.epsilon=.5;
PeakOpts.DiffusionOpts.K=10;
PeakOpts.DiffusionTime=100;
%}

% DataSet='SyntheticSpherical';
% M=0;
% N=0;
% Budget=[1,5:5:65];
% PeakOpts.DiffusionOpts.epsilon=.5;
% PeakOpts.DiffusionOpts.K=10;
% PeakOpts.DiffusionTime=100;

% {
% DataSet='SyntheticGeometric';
% M=0;
% N=0;
% Budget=1:1:20;
% PeakOpts.DiffusionOpts.epsilon=.15;
% PeakOpts.DiffusionOpts.K=10;
% PeakOpts.DiffusionTime=10^4.5;
% }

% {
% DataSet='Pavia';
% M=60;
% N=300;
% Budget=10:10:500;
% PeakOpts.DiffusionOpts.epsilon=1;
% PeakOpts.DiffusionOpts.K='automatic';
% PeakOpts.DiffusionTime=30;
% }

% % {
DataSet='SalinasA';
M=83;
N=86;
Budget = [10 20 100 200:200:2000 ];
PeakOpts.DiffusionOpts.epsilon=1;
PeakOpts.DiffusionOpts.K='automatic';
PeakOpts.DiffusionTime=30;
% % }
% Prepare data based on VAE active learning
X_VAE = csvread('salinas_train_dim40.csv');
X_VAE = X_VAE./sqrt(sum(X_VAE.*X_VAE,2));
LabelsGT_VAE = csvread('salinas_train_label_dim40.csv');
K_GT = 6;

LabelsGT_VAE= LabelsGT_VAE';
PeakOpts.DiffusionOpts.epsilon=1;
PeakOpts.DiffusionOpts.K='automatic';
PeakOpts.DiffusionTime= 30;
PeakOpts_VAE.DiffusionOpts.epsilon=1;
PeakOpts_VAE.DiffusionOpts.K='automatic';
PeakOpts_VAE.DiffusionTime= 30;
%% Set parameters

[X,LabelsGT,K_GT]=ExperimentalData(DataSet);
PeakOpts.UserPlot=0;
PeakOpts_VAE.UserPlot=0;

%bHow many nearest neighbors to use for diffusion distance
PeakOpts_VAE.DiffusionOpts.kNN=30; 
PeakOpts.DiffusionOpts.kNN=100; 

%bForce probability of self-loop to exceed .5.
PeakOpts.DiffusionOpts.LazyWalk=0;
PeakOpts_VAE.DiffusionOpts.LazyWalk=0;

% How many nearest neighbors to use for KDE
DensityNN=20;
DensityNN_VAE = 100;

% Mode detection
PeakOpts.ModeDetection='Diffusion';
PeakOpts_VAE.ModeDetection='Diffusion';

%% Find densities, diffusion distances, and data modes

[Centers, G, DistStruct] = DensityPeaksEstimation(X, K_GT, DensityNN, PeakOpts);
[Centers_VAE, G_VAE, DistStruct_VAE] = DensityPeaksEstimation(X_VAE, K_GT, DensityNN_VAE, PeakOpts_VAE);

%%  Compute active learning queries

CandidateQueries=find(LabelsGT>0);
[~,Idx]=sort(DistStruct.DeltaDensity(CandidateQueries),'descend');
CandidateQueries=CandidateQueries(Idx);

CandidateQueries_VAE=find(LabelsGT_VAE>0);
[~,Idx_V]=sort(DistStruct_VAE.DeltaDensity(CandidateQueries_VAE),'descend');
CandidateQueries_VAE=CandidateQueries_VAE(Idx_V);

Queries_LAND=CandidateQueries;
Queries_Random=CandidateQueries(randperm(length(CandidateQueries)));
Queries_LAND_VAE=CandidateQueries_VAE;
Queries_Random_VAE=CandidateQueries_VAE(randperm(length(CandidateQueries_VAE)));
length(Queries_LAND)
%%  Compute LAND labels
for k=1:length(Budget)
    k
    LabelsLAND=LAND(X,K_GT,DistStruct,Centers,LabelsGT,Queries_LAND(1:Budget(k)));
    LabelsRandom=LAND(X,K_GT,DistStruct,Centers,LabelsGT,Queries_Random(1:Budget(k)));
    [OA_LAND(k)] = GetAccuracies(LabelsLAND(LabelsGT>0),UniqueGT(LabelsGT(LabelsGT>0)),K_GT);
    [OA_Random(k)] = GetAccuracies(LabelsRandom(LabelsGT>0),UniqueGT(LabelsGT(LabelsGT>0)),K_GT);
end

for k=1:length(Budget)
    k
    LabelsLAND_VAE=LAND(X_VAE,K_GT,DistStruct_VAE,Centers_VAE,LabelsGT_VAE,Queries_LAND_VAE(1:Budget(k)));
    LabelsRandom_VAE=LAND(X_VAE,K_GT,DistStruct_VAE,Centers_VAE,LabelsGT_VAE,Queries_Random_VAE(1:Budget(k)));
    [OA_LAND_VAE(k)] = GetAccuracies(LabelsLAND_VAE(LabelsGT_VAE>0),UniqueGT(LabelsGT_VAE(LabelsGT_VAE>0)),K_GT);
    [OA_Random_VAE(k)] = GetAccuracies(LabelsRandom_VAE(LabelsGT_VAE>0),UniqueGT(LabelsGT_VAE(LabelsGT_VAE>0)),K_GT);
end

%%  Plot HSI data, GT, Results

if M>0 && N>0
    
    figure;
    imagesc(reshape(LabelsGT,M,N));
    axis equal
    axis off;
    axis tight;
    
end
f= figure;
set(gcf,'renderer','Painters')
plot(OA_LAND,'LineWidth',3,'Color','r');
hold on;
plot(OA_LAND_VAE,'LineWidth',3,'Color','b');
hold on;
plot(OA_Random,'--','LineWidth',3,'Color','r');
hold on;
plot(OA_Random_VAE,'--','LineWidth',3, 'Color','b');


lgd=legend('OA, LAND','OA, VAE-LAND','OA, LAND-Random ','OA, VAE-LAND Random ',...
    'Interpreter','latex');
set(lgd,'location','best')
axis([1 length(Budget) min(vertcat(OA_LAND(:),OA_LAND_VAE(:), OA_Random(:),OA_Random_VAE(:)))-.05 1])
xticks([1,2:1:length(Budget)])
xticklabels(Budget([1,2:1:length(Budget)]))
title('Performance of active learning algorithms','Interpreter','latex','FontSize',14)
xlabel('Number Queries','Interpreter','latex','FontSize',14);
ylabel('Accuracy','Interpreter','latex','FontSize',14);
