%
% matlab program for boosted constrained k-means
%
function K_new = boosted_constrained_kmeans(K,cnum,C,T);
% K: initial kernel matrix (e.g. calculated by linear kernel or rbf kernel)
% cnum: number of clusters
% C: constraint matrix (dnum x dnum), dnum is the number of data
%    C(i,j) = 1 or -1 depending on a data pair (i,j) is a must-link or cannot-link
% T: number of boosting steps

% number of data
dnum = length(K(1,:));

% constrained pair and its weight (ci,cj, m/c or 1/-1, wt)
cset = [];
[ cset(:,1), cset(:,2), cset(:,3) ] = find(C);

% initialize the weight(priority) of each constraint
n_constr = length(cset(:,1));
for i=1:n_constr
  cset(i,4) = 1.0/n_constr;
end
        
% boosted kernel matrix
K_new = sparse(dnum,dnum);

% start boosting
for t=1:T
  t
  % run constrained kmeans
  % ass = dnum x cnum matrix
  %   if ass(i,j) == 1 then data i is assigned to cluster j
  ass = constrained_kmeans(K,cnum,cset);
            
  % calc error rate
  err = 0.0;
  for i=1:n_constr
    h = 2*ass(cset(i,1),:)*ass(cset(i,2),:)' - 1;
    if h*cset(i,3) < 0
      err = err + cset(i,4);
    end
  end
  err = err / sum(cset(:,4));
  
  % update boosted kernel matrix depending on the error rate
  alp = 0.0;
  if err == 0
    alp = 100;
    K_new = K_new + alp*ass*ass';
    break;
  elseif 0 < err && err < 0.5
    alp = 0.5*log( (1-err)/err );
    K_new = K_new + alp*ass*ass';
  else % err >= 0.5
    break;
  end
            
  % update weights of constraints
  for i=1:n_constr
    h = 2*ass(cset(i,1),:)*ass(cset(i,2),:)' - 1;
    cset(i,4) = cset(i,4)*exp(-alp*cset(i,3)*h);
  end
            
end
