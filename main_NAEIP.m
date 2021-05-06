%----------------------------
% NAE-IP
% using synthetic data
%----------------------------
% by Ayano NAKAI-KASAI (nakai.ayano@sys.i.kyoto-u.ac.jp)

clear;
close all;

%--- Set path to GPML Toolbox ---%
% addpath(genpath('~~~'));

%--- Parameters for data settings ---%
ntrain = 10^4;              % number of training data points
dtrain = 1;                 % dimension of training data
ntest = ntrain*10^(-2);     % number of test data
dtest = 1;                  % dimension of test data
noisevar = 0.04;            % additive noise ~ Gaussian(0,noisevar)
rtrain = 4;                 % range of training data
rtest = 5;                  % range of test data
nIteration = 10;            % number of iteration
%--- Parameters for NAE-IP ---%
ni = 500;                   % number of subdata per each expert
p = ntrain/ni;              % number of experts
nt = 50;                    % number of test inputs processed at once
scale = 1.5;                % scaling factor for sketching dimension, nu=scale*nipt
f = @(x) sinc(x);
%--- Initialization ---%
t_NAEIPbt = zeros(nIteration,1);
mse_NAEIPbt = zeros(nIteration,1); msll_NAEIPbt = zeros(nIteration,1);
t_NAEIPbtot = zeros(nIteration,1);
mse_NAEIPbtot = zeros(nIteration,1); msll_NAEIPbtot = zeros(nIteration,1);
t_NAEIPbtnt = zeros(nIteration,1);
mse_NAEIPbtnt = zeros(nIteration,1); msll_NAEIPbtnt = zeros(nIteration,1);
t_NAEIPat = zeros(nIteration,1);
mse_NAEIPat = zeros(nIteration,1); msll_NAEIPat = zeros(nIteration,1);
t_NAEIPnt = zeros(nIteration,1);
mse_NAEIPnt = zeros(nIteration,1); msll_NAEIPnt = zeros(nIteration,1);

for itr = 1:nIteration
    disp(['Iteration = ' num2str(itr)]);
    %--- Generation of training and test data ---%
    xtrain = linspace(-rtrain,rtrain,ntrain)';                  % training samples in [-rtrain,rtrain]
    ztrain = f(xtrain) + sqrt(noisevar).*randn(ntrain,dtrain);  % observations
    xtest = -rtest + 2*rtest*rand(ntest,dtest);                 % test samples in [-ranges,ranges]
    xtest = sort(xtest,'ascend');                               % for simplicity, transform xtest to ascend-ordered
    ztest = f(xtest) + sqrt(noisevar).*randn(ntest,dtest);      % targets
    %- normalization
    xtrain_mean = mean(xtrain); 
    xtrain_std  = std(xtrain);
    xtrain = (xtrain-repmat(xtrain_mean,ntrain,1)) ./ repmat(xtrain_std,ntrain,1) ;    
    ztrain_mean = mean(ztrain);
    ztrain_std  = std(ztrain);
    ztrain = (ztrain-ztrain_mean)/ztrain_std;
    xtest = (xtest - repmat(xtrain_mean,ntest,1)) ./ (repmat(xtrain_std,ntest,1));
    
    %- model parameters
    sf = 0.25;
    ell = 0.8;
    sn = 0.5; 
    meanfunc = [];
    covfunc = @covSEiso;                % squared exponential covariance function
    likfunc = @likGauss;                % Gaussian likelihood
    inffunc = @infGaussLik;
    hyp = struct('mean', meanfunc, 'cov', [log(ell) log(sf)], 'lik', log(sn));
    
    %- sub-model construction
    models = cell(1,p);
    Idx = randperm(ntrain);             % random partition
    for i = 1:p
        Idxi = Idx((i-1)*ni+1:i*ni);
        models{i}.Xi = xtrain(Idxi);
        models{i}.Zi = ztrain(Idxi);
        models{i}.Ximean = mean(models{i}.Xi);
        models{i}.hyp = hyp;
        models{i}.meanfunc = meanfunc;
        models{i}.likfunc = likfunc;
        models{i}.covfunc = covfunc;
        models{i}.inffunc = inffunc;
    end
    
    criterion = 'NAEIPbt' ; % NAEIPbt, NAEIPbtot, NAEIPbtnt, NAEIPat, NAEIPnt
    [mu_NAEIPbt,s2_NAEIPbt,t_NAEIPbt(itr)] = prediction(xtest,models,criterion,nt);
    mu_NAEIPbt = mu_NAEIPbt*ztrain_std + ztrain_mean;
    s2_NAEIPbt = s2_NAEIPbt*ztrain_std^2;
    [mse_NAEIPbt(itr),msll_NAEIPbt(itr)] = Results(ztest,mu_NAEIPbt,s2_NAEIPbt);
    disp(['NAEIPbt: ' num2str(mse_NAEIPbt(itr)) ', ' num2str(msll_NAEIPbt(itr)) ', ' num2str(t_NAEIPbt(itr))]);
    
    criterion = 'NAEIPbtot' ; % NAEIPbt, NAEIPbtot, NAEIPbtnt, NAEIPat, NAEIPnt
    nip = scale*nt - nt;            % number of inducing points except for test inputs
    [mu_NAEIPbtot,s2_NAEIPbtot,t_NAEIPbtot(itr)] = prediction(xtest,models,criterion,nt,nip);
    mu_NAEIPbtot = mu_NAEIPbtot*ztrain_std + ztrain_mean;
    s2_NAEIPbtot = s2_NAEIPbtot*ztrain_std^2;
    [mse_NAEIPbtot(itr),msll_NAEIPbtot(itr)] = Results(ztest,mu_NAEIPbtot,s2_NAEIPbtot);
    disp(['NAEIPbtot: ' num2str(mse_NAEIPbtot(itr)) ', ' num2str(msll_NAEIPbtot(itr)) ', ' num2str(t_NAEIPbtot(itr))]);
    
    criterion = 'NAEIPbtnt' ; % NAEIPbt, NAEIPbtot, NAEIPbtnt, NAEIPat, NAEIPnt
    nip = scale*nt - nt;            % number of inducing points except for test inputs
    for i = 1:p
        Xip{i} = repmat(models{i}.Ximean,nip,1) + repmat(std(models{i}.Xi),nip,1).*randn(nip,dtrain);
    end
    [mu_NAEIPbtnt,s2_NAEIPbtnt,t_NAEIPbtnt(itr)] = prediction(xtest,models,criterion,nt,nip,Xip);
    mu_NAEIPbtnt = mu_NAEIPbtnt*ztrain_std + ztrain_mean;
    s2_NAEIPbtnt = s2_NAEIPbtnt*ztrain_std^2;
    [mse_NAEIPbtnt(itr),msll_NAEIPbtnt(itr)] = Results(ztest,mu_NAEIPbtnt,s2_NAEIPbtnt);
    disp(['NAEIPbtnt: ' num2str(mse_NAEIPbtnt(itr)) ', ' num2str(msll_NAEIPbtnt(itr)) ', ' num2str(t_NAEIPbtnt(itr))]);

    criterion = 'NAEIPat' ; % NAEIPbt, NAEIPbtot, NAEIPbtnt, NAEIPat, NAEIPnt
    nip = scale*nt;                 % number of inducing points
    [mu_NAEIPat,s2_NAEIPat,t_NAEIPat(itr)] = prediction(xtest,models,criterion,nt,nip);
    mu_NAEIPat = mu_NAEIPat*ztrain_std + ztrain_mean;
    s2_NAEIPat = s2_NAEIPat*ztrain_std^2;
    [mse_NAEIPat(itr),msll_NAEIPat(itr)] = Results(ztest,mu_NAEIPat,s2_NAEIPat);
    disp(['NAEIPat: ' num2str(mse_NAEIPat(itr)) ', ' num2str(msll_NAEIPat(itr)) ', ' num2str(t_NAEIPat(itr))]);
    
    criterion = 'NAEIPnt' ; % NAEIPbt, NAEIPbtot, NAEIPbtnt, NAEIPat, NAEIPnt
    nip = scale*nt;                 % number of inducing points
    for i = 1:p
        Xip{i} = repmat(models{i}.Ximean,nip,1) + repmat(std(models{i}.Xi),nip,1).*randn(nip,dtrain);
    end
    [mu_NAEIPnt,s2_NAEIPnt,t_NAEIPnt(itr)] = prediction(xtest,models,criterion,nt,nip,Xip);
    mu_NAEIPnt = mu_NAEIPnt*ztrain_std + ztrain_mean;
    s2_NAEIPnt = s2_NAEIPnt*ztrain_std^2;
    [mse_NAEIPnt(itr),msll_NAEIPnt(itr)] = Results(ztest,mu_NAEIPnt,s2_NAEIPnt);
    disp(['NAEIPnt: ' num2str(mse_NAEIPnt(itr)) ', ' num2str(msll_NAEIPnt(itr)) ', ' num2str(t_NAEIPnt(itr))]);

end


function [mse,msll] = Results(ztest,mu,s2)

mse = mean((ztest-mu).^2);
msll = mean(log(2*pi().*s2)./2 + (ztest-mu).^2./(2.*s2));

end