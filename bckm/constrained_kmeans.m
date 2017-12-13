%---- run constrained kernel kmeans ----%
function ass = constrained_kmeans(K,cnum,cset);
% ass = dnum x cnum matrix
%   if ass(i,j) == 1 then data i is assigned to cluster j
% K: initial kernel matrix (e.g. calculated by linear kernel or rbf
% kernel)
% cnum: number of clusters
% cset = find(C)
%    C: constraint matrix (dnum x dnum), dnum is the number of data
%       C(i,j) = 1 or -1 depending on a data pair (i,j) is a must-link or cannot-link

% number of data
dnum = length(K(1,:));

% sort constraints by weight
[vsort_cset,isort_cset] = sort(cset(:,4)','descend');
  
% matrix (dnum x cnum) for cluster assignment
ass = sparse(dnum,cnum);

% initial seeds by k-means++
n = 1;
seed = [];
seed(n) = ceil(rand * dnum);
while n < cnum
  % calc distance to the nearest seed
  d2 = diag(K)*ones(1,n) - 2*K(:,[seed]) + ones(dnum,n)*diag(diag(K([seed],[seed])));
  if n > 1
    [v_mm i_mm] = max(min(d2'));
  else
    [v_mm i_mm] = max(d2');
  end
  n = n + 1;
  seed(n) = i_mm;
end

% initial cluster assignment
for i=1:length(seed)
    ass(seed(i),i) = 1;
end

% loop for kmeans
v_obj = 1.0e+10;
sum1 = zeros(dnum,cnum); % k(xi,xj)
sum2 = zeros(1,cnum);    % k(sigma(xi),sigma(xi))
for n_loop=1:100
    
    % calc distance to each center
    for i=1:cnum
        mem = find(ass(:,i));
        n_mem = length(mem);
        if n_mem > 0
            sum1(:,i) = -2*sum(K(:,[mem']),2)/n_mem;
            sum2(1,i) = -0.5*sum(sum1([mem],i))/n_mem;
        end
    end

    % calc nearest centroid
    [vsort,isort] = sort( (sum1(:,:)+ones(dnum,1)*sum2(1,:))' );
    vsort = vsort';
    isort = isort';
    
    % refresh data assignment
    v_obj_prev = v_obj;
    v_obj = 0;
    ass_prev = ass;
    ass = sparse(dnum,cnum);
    uncons = ones(1,dnum); % check for unconstrained data
    % 1st stage assign data with constraints
    for i=isort_cset
        c1 = cset(i,1);
        c2 = cset(i,2);
        cv = cset(i,3);
        
        % if both are not assigned
        if uncons(c1)==1 && uncons(c2)==1
            if cv == 1 % must-link
                if vsort(c1,1) < vsort(c2,1)
                    c_opt = isort(c1,1);
                else
                    c_opt = isort(c2,1);
                end
                ass(c1,c_opt) = 1;
                ass(c2,c_opt) = 1;
                
            else % cannot-link
                c_c1 = isort(c1,1);
                c_c2 = isort(c2,1);
                if c_c1 == c_c2
                    if vsort(c1,1) < vsort(c2,1)
                        c_c2 = isort(c2,2);
                    else
                        c_c1 = isort(c1,2);
                    end
                end
                ass(c1,c_c1) = 1;
                ass(c2,c_c2) = 1;
                
            end
            uncons(1,c1) = 0;
            uncons(1,c2) = 0;
            
        elseif uncons(c1)==0 && uncons(c2)==1 % if only c1 is assigned
            % c1's nearest centroid
            c_c1 = find(ass(c1,:));
            
            if cv == 1 %must-link
                ass(c2,c_c1) = 1;
                
            else %cannot-link
                for c=isort(c2,:)
                    if c ~= c_c1
                        ass(c2,c) = 1;
                        break
                    end
                end
            end
            uncons(1,c2) = 0;
            
        elseif uncons(c1)==1 && uncons(c2)==0 % if only c2 is assigned
            % c2's nearest centroid
            c_c2 = find(ass(c2,:));
            
            if cv == 1 % must-link
                ass(c1,c_c2) = 1;
                
            else %cannot-link
                for c=isort(c1,:)
                    if c ~= c_c2
                        ass(c1,c) = 1;
                        break
                    end
                end
                
            end
            uncons(1,c1) = 0;
            
        elseif uncons(c1)==0 && uncons(c2)==0
            % do nothing
        end
    end
    
    % secondly, assign unconstrained data
    for i=find(uncons(1,:))
        [d_min,i_min] = min( sum1(i,:)+sum2(1,:) );
        ass(i,i_min) = 1;
    end

    v_obj = sum(diag(K));
    for i=1:cnum
        mem = find(ass(:,i));
        n_mem = length(mem);
        v_obj = v_obj - sum(sum(K([mem'],[mem])))/n_mem;
    end
    
    if v_obj >= v_obj_prev
        ass = ass_prev;
        break
    end
    
    if isequal(ass,ass_prev)
        break
    end
    
end
