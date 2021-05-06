function [mu,s2,t_predict] = prediction(xtest,models,criterion,varargin)
% Prediction by NAE-IP
% Inputs:
%  Xt: test points
%  models: p submodels
%  criterion: NAE-IP with 5 options
%             'NAEIPbt': blockwise test points
%             'NAEIPnt': non-test points
%             'NAEIPbtnt': blockwise test points + non-test points
%             'NAEIPbtot': blockwise test points + other test points
%             'NAEIPat': arbitrary test points
%  varargin: variables for NAE-IP
%            nt: number of test inputs processing at once, for NAEIPbt
%            nip: number of inducing points, for NAEIPbtot and NAEIPat
%            Xip: p cells of inducing points, for NAEIPnt and NAEIPbtnt
% Outputs:
%  mu: predictive mean
%  s2: predictive variance
%  t_predict: computing time for predictions
%
% by Ayano NAKAI-KASAI (nakai.ayano@sys.i.kyoto-u.ac.jp).
% Some parts of functions are based on the implementation by Haitao Liu (H.
% Liu et al., ICML 2018; https://github.com/LiuHaiTao01/GRBCM)

if nargin==6   % NAEIPnt and NAEIPbtnt
    nt = varargin{1};
    nip = varargin{2};
    Xip = varargin{3};
elseif nargin==5   % NAEIPbtot and NAEIPat
    nt = varargin{1};
    nip = varargin{2};
elseif nargin==4   % NAEIPbt
    nt = varargin{1};
end   % others

ntest = size(xtest,1);          % number of test points
p = length(models);             % number of experts
hyp_lik = models{1}.hyp.lik;

%--- Start prediction ---%
t1 = cputime;
if iscell(models{1}.covfunc)
    covfuncm = models{1}.covfunc;
    ktt = feval(covfuncm{:},models{1}.hyp.cov,xtest,'diag');    % K**
else
    ktt = feval(models{1}.covfunc,models{1}.hyp.cov,xtest,'diag');
end
K_invs = inverseKernelMarix_submodels(models);                  % Kii^{-1}
K_cross = crossKernelMatrix_nestedKG(models);                   % Kij

mu = zeros(ntest,1); s2 = zeros(ntest,1);
switch criterion  
    case 'NAEIPbt'
        for i = 1:p   
            [mui{i},~] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
                                       models{i}.covfunc,models{i}.likfunc,models{i}.Xi,models{i}.Zi,xtest);
        end
        Mu = zeros(ntest,p);
        for i = 1:p, Mu(:,i) = mui{i}; end

        Numt = ceil(ntest/nt);
        for i = 1:Numt
            if i*nt > ntest
                Idx = (i-1)*nt+1:ntest;  % remainder of test inputs
            else
                Idx = (i-1)*nt+1:i*nt;
            end
            xt = xtest(Idx,:);
            tmpMx = Mu(Idx,:);
            MuA = tmpMx(:);
            kA = kernelVector_nestedKGbt(xt,models,K_invs);
            [~,KA_inv] = kernelMatrix_nestedKGbt(xt,models,K_invs,K_cross);

            mu(Idx) = kA'*KA_inv*MuA;
            s2(Idx) = ktt(Idx) - diag(kA'*KA_inv*kA) + exp(2*hyp_lik);
        end
    case 'NAEIPbtot'
        Numt = ceil(ntest/nt);
        for i = 1:Numt
            if i*nt > ntest
                Idx = (i-1)*nt+1:ntest;  % remainder of test inputs
            else
                Idx = (i-1)*nt+1:i*nt;
            end
            nu = nip + length(Idx);         % dimension
            xt = xtest(Idx,:);
            tmpXt = xtest;
            tmpXt(Idx,:) = [];              % test points except for xt
            MuA = zeros(nu*p,1);
            for j = 1:p   
                Idx2 = randperm(size(tmpXt,1),nip);
                Xipot{j} = tmpXt(Idx2,:);   % other test points
                [MuA((j-1)*nu+1:j*nu),~] = gp(models{j}.hyp,models{j}.inffunc,models{j}.meanfunc, ...
                                       models{j}.covfunc,models{j}.likfunc,models{j}.Xi,models{j}.Zi,[xt;Xipot{j}]);
            end
            kA = kernelVector_nestedKG_NAEIPbtnt(xt,models,K_invs,Xipot);
            [~,KA_inv] = kernelMatrix_nestedKG_NAEIPbtnt(xt,models,K_invs,K_cross,Xipot);

            mu(Idx) = kA'*KA_inv*MuA;
            s2(Idx) = ktt(Idx) - diag(kA'*KA_inv*kA) + exp(2*hyp_lik);
        end
    case 'NAEIPbtnt'
        Numt = ceil(ntest/nt);
        for i = 1:Numt
            if i*nt > ntest
                Idx = (i-1)*nt+1:ntest;  % remainder of test inputs
            else
                Idx = (i-1)*nt+1:i*nt;
            end
            nu = nip + length(Idx);         % dimension
            xt = xtest(Idx,:);
            MuA = zeros(nu*p,1);
            for j = 1:p   
                [MuA((j-1)*nu+1:j*nu),~] = gp(models{j}.hyp,models{j}.inffunc,models{j}.meanfunc, ...
                                       models{j}.covfunc,models{j}.likfunc,models{j}.Xi,models{j}.Zi,[xt;Xip{j}]);
            end
            kA = kernelVector_nestedKG_NAEIPbtnt(xt,models,K_invs,Xip);
            [~,KA_inv] = kernelMatrix_nestedKG_NAEIPbtnt(xt,models,K_invs,K_cross,Xip);

            mu(Idx) = kA'*KA_inv*MuA;
            s2(Idx) = ktt(Idx) - diag(kA'*KA_inv*kA) + exp(2*hyp_lik);
        end
    case 'NAEIPat' % random test inputs
        nu = nip;                       % dimension
        MuA = zeros(nu*p,1);
        for i = 1:p
            Idx2 = randperm(ntest,nip);
            Xipat{i} = xtest(Idx2,:);   % arbitrary test points
            [MuA((i-1)*nu+1:i*nu),~] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
                                   models{i}.covfunc,models{i}.likfunc,models{i}.Xi,models{i}.Zi,Xipat{i});
        end
        [~,KA_inv] = kernelMatrix_nestedKG_NAEIPnt(models,K_invs,K_cross,Xipat);
        Numt = ceil(ntest/nt);
        for i = 1:Numt
            if i*nt > ntest
                Idx = (i-1)*nt+1:ntest;  % remainder of test inputs
            else
                Idx = (i-1)*nt+1:i*nt;
            end
            xt = xtest(Idx,:);
            kA = kernelVector_nestedKG_NAEIPnt(xt,models,K_invs,Xipat);

            mu(Idx) = kA'*KA_inv*MuA;
            s2(Idx) = ktt(Idx) - diag(kA'*KA_inv*kA) + exp(2*hyp_lik);
        end
    case 'NAEIPnt'
        for i = 1:p 
            [mui{i},~] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
                                       models{i}.covfunc,models{i}.likfunc,models{i}.Xi,models{i}.Zi,Xip{i});
        end
        nu = nip;                       % dimension
        MuA = zeros(nu*p,1);
        for i = 1:p, MuA((i-1)*nip+1:i*nip) = mui{i}; end
        [~,KA_inv] = kernelMatrix_nestedKG_NAEIPnt(models,K_invs,K_cross,Xip);

        Numt = ceil(ntest/nt);
        for i = 1:Numt
            if i*nt > ntest
                Idx = (i-1)*nt+1:ntest; % remainder of test inputs
            else
                Idx = (i-1)*nt+1:i*nt;
            end
            xt = xtest(Idx,:);
            kA = kernelVector_nestedKG_NAEIPnt(xt,models,K_invs,Xip);

            mu(Idx) = kA'*KA_inv*MuA;
            s2(Idx) = ktt(Idx) - diag(kA'*KA_inv*kA) + exp(2*hyp_lik);
        end
    otherwise
        error('Use NAEIPbt, NAEIPbtot, NAEIPbtnt, NAEIPat, or NAEIPnt.');
end

t_predict = cputime - t1;

end


%%%%%%%%%%%%%%%
function [K_invs] = inverseKernelMarix_submodels(models)
% calculate the covariance matrics Ks, the inverse matrics K_invs and the det of matrics K_dets of submodels
% used for the nestedKG criterion
p = length(models);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov; hyp_lik = models{1}.hyp.lik;
for i = 1:p
    if iscell(covfunc)
        K_Xi = feval(covfunc{:},hyp_cov,models{i}.Xi) + exp(2*hyp_lik)*eye(size(models{i}.Xi,1));
    else
        K_Xi = feval(covfunc,hyp_cov,models{i}.Xi) + exp(2*hyp_lik)*eye(size(models{i}.Xi,1));
    end
    K_invs{i} = eye(size(models{i}.Xi,1))/K_Xi;
end

end % end function


function K_cross = crossKernelMatrix_nestedKG(models)
% construct the covariance of training points
% used for the nestedKG criterion 
p = length(models);
K_cross = cell(p,p);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov; hyp_lik = models{1}.hyp.lik;
for i = 1:p 
    for j = 1:p 
        if iscell(covfunc)
            if i == j % self-covariance, should consider noise term
                K_cross{i}{j} = feval(covfunc{:},hyp_cov,models{i}.Xi,models{j}.Xi) + exp(2*hyp_lik)*eye(size(models{i}.Xi,1));
            else % cross-covariance
                K_cross{i}{j} = feval(covfunc{:},hyp_cov,models{i}.Xi,models{j}.Xi);
            end
        else
            if i == j % self-covariance, should consider noise term
                K_cross{i}{j} = feval(covfunc,hyp_cov,models{i}.Xi,models{j}.Xi) + exp(2*hyp_lik)*eye(size(models{i}.Xi,1));
            else % cross-covariance
                K_cross{i}{j} = feval(covfunc,hyp_cov,models{i}.Xi,models{j}.Xi);
            end
        end 
    end
end

end % end function


function kA = kernelVector_nestedKGbt(x,models,K_invs)
% construct the covariance between test points and training points
% used for the nestedKG criterion 
p = length(models);
nt = size(x,1);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov;
kA = zeros(nt*p,nt);
for i = 1:p 
    if iscell(covfunc)
        k_x_Xi = feval(covfunc{:},hyp_cov,x,models{i}.Xi);
    else
        k_x_Xi = feval(covfunc,hyp_cov,x,models{i}.Xi);
    end
    kA((i-1)*nt+1:i*nt,:) = k_x_Xi*K_invs{i}*k_x_Xi';
end

end % end function


function [KA,KA_inv] = kernelMatrix_nestedKGbt(x,models,K_invs,K_cross)
% construct the covariance of training points
% used for the nestedKG criterion 
p = length(models);
nipt = size(x,1);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov;
KA = zeros(nipt*p,nipt*p);
for i = 1:p 
    if iscell(covfunc)
        k_x_Xi = feval(covfunc{:},hyp_cov,x,models{i}.Xi);
    else
        k_x_Xi = feval(covfunc,hyp_cov,x,models{i}.Xi);
    end
    KA((i-1)*nipt+1:i*nipt,(i-1)*nipt+1:i*nipt) = 0.5.*k_x_Xi*K_invs{i}*k_x_Xi'; % the coef 0.5 is used to ensure KA = KA + KA' along diagonal line
    for j = i+1:p 
        if iscell(covfunc)
            k_Xj_x = feval(covfunc{:},hyp_cov,models{j}.Xi,x);
        else
            k_Xj_x = feval(covfunc,hyp_cov,models{j}.Xi,x);
        end
        K_Xi_Xj = K_cross{i}{j};
        KA((i-1)*nipt+1:i*nipt,(j-1)*nipt+1:j*nipt) = k_x_Xi*K_invs{i}*K_Xi_Xj*K_invs{j}*k_Xj_x;
    end
end
% obtain whole KA
KA = KA + KA';

jitter = 1e-10;
KA_inv = eye(size(KA,1))/(KA + jitter*eye(size(KA,1)));

end % end function


function [KA,KA_inv] = kernelMatrix_nestedKG_NAEIPnt(models,K_invs,K_cross,Xip)
% construct the covariance of inducing points
% used for the nestedKG criterion 
p = length(models);
nip = size(Xip{1},1);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov;
KA = zeros(nip*p,nip*p);
% obtain an upper triangular matrix to save compting time
for i = 1:p 
    if iscell(covfunc)
        k_xipi_Xi = feval(covfunc{:},hyp_cov,Xip{i},models{i}.Xi);
    else
        k_xipi_Xi = feval(covfunc,hyp_cov,Xip{i},models{i}.Xi);
    end
    KA((i-1)*nip+1:i*nip,(i-1)*nip+1:i*nip) = 0.5*k_xipi_Xi*K_invs{i}*k_xipi_Xi'; % the coef 0.5 is used to ensure KA = KA + KA' along diagonal line
    for j = i+1:p 
        if iscell(covfunc)
            k_Xj_xipj = feval(covfunc{:},hyp_cov,models{j}.Xi,Xip{j});
        else
            k_Xj_xipj = feval(covfunc,hyp_cov,models{j}.Xi,Xip{j});
        end
        KA((i-1)*nip+1:i*nip,(j-1)*nip+1:j*nip) = k_xipi_Xi*K_invs{i}*K_cross{i}{j}*K_invs{j}*k_Xj_xipj;
    end
end
% obtain whole KA
KA = KA + KA';

jitter = 1e-10;
KA_inv = eye(size(KA,1))/(KA + jitter*eye(nip*p));

end % end function


function [KA,KA_inv] = kernelMatrix_nestedKG_NAEIPbtnt(x,models,K_invs,K_cross,Xip)
% construct the covariance of test + other points
% used for the nestedKG criterion 
p = length(models);
nip = size(Xip{1},1)+size(x,1);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov;
KA = zeros(nip*p,nip*p);
% obtain an upper triangular matrix to save compting time
for i = 1:p
    if iscell(covfunc)
        k_xipi_Xi = feval(covfunc{:},hyp_cov,[x;Xip{i}],models{i}.Xi);
    else
        k_xipi_Xi = feval(covfunc,hyp_cov,[x;Xip{i}],models{i}.Xi);
    end
    KA((i-1)*nip+1:i*nip,(i-1)*nip+1:i*nip) = 0.5*k_xipi_Xi*K_invs{i}*k_xipi_Xi'; % the coef 0.5 is used to ensure KA = KA + KA' along diagonal line
    for j = i+1:p
        if iscell(covfunc)
            k_Xj_xipj = feval(covfunc{:},hyp_cov,models{j}.Xi,[x;Xip{j}]);
        else
            k_Xj_xipj = feval(covfunc,hyp_cov,models{j}.Xi,[x;Xip{j}]);
        end
        KA((i-1)*nip+1:i*nip,(j-1)*nip+1:j*nip) = k_xipi_Xi*K_invs{i}*K_cross{i}{j}*K_invs{j}*k_Xj_xipj;
    end
end
% obtain whole KA
KA = KA + KA';

jitter = 1e-10;
KA_inv = eye(size(KA,1))/(KA + jitter*eye(nip*p));

end % end function


function kA = kernelVector_nestedKG_NAEIPnt(x,models,K_invs,Xip)
% construct the covariance between inducing points and training points
% used for the nestedKG criterion 
p = length(models);
nip = size(Xip{1},1);
nt = size(x,1);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov;
kA = zeros(nip*p,nt);
for i = 1:p 
    if iscell(covfunc)
        k_xipi_Xi = feval(covfunc{:},hyp_cov,Xip{i},models{i}.Xi);
        k_Xi_xt = feval(covfunc{:},hyp_cov,models{i}.Xi,x);
    else
        k_xipi_Xi = feval(covfunc,hyp_cov,Xip{i},models{i}.Xi);
        k_Xi_xt = feval(covfunc,hyp_cov,models{i}.Xi,x);
    end
    kA((i-1)*nip+1:i*nip,:) = k_xipi_Xi*K_invs{i}*k_Xi_xt;
end

end % end function


function kA = kernelVector_nestedKG_NAEIPbtnt(x,models,K_invs,Xip)
% construct the covariance between test + other points and training points
% used for the nestedKG criterion 
p = length(models);
nip = size(Xip{1},1)+size(x,1);
nt = size(x,1);

covfunc = models{1}.covfunc; hyp_cov = models{1}.hyp.cov;
kA = zeros(nip*p,nt);
for i = 1:p 
    if iscell(covfunc)
        k_xipi_Xi = feval(covfunc{:},hyp_cov,[x;Xip{i}],models{i}.Xi);
        k_Xi_xt = feval(covfunc{:},hyp_cov,models{i}.Xi,x);
    else
        k_xipi_Xi = feval(covfunc,hyp_cov,[x;Xip{i}],models{i}.Xi);
        k_Xi_xt = feval(covfunc,hyp_cov,models{i}.Xi,x);
    end
    kA((i-1)*nip+1:i*nip,:) = k_xipi_Xi*K_invs{i}*k_Xi_xt;
end

end % end function


