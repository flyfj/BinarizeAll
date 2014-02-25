  
load cifar_split;
[n,d] = size(traindata);
tn = size(testdata,1);
range = 100; 
r = 48;


%% IsoHash
tic;
mvec = mean(traindata,1);
traindata = traindata-repmat(mvec,n,1);
cov = traindata'*traindata;
[U,V] = eig(cov);
eigenvalue = diag(V)';
[eigenvalue,order] = sort(eigenvalue,'descend');
W = U(:,order(1:r));
eigenvalue = eigenvalue(1:r);

R = GradientFlow(diag(eigenvalue));
W = W*R;
mvec = mvec*W;
Y = traindata*W;
Y = (Y>0);
B = compactbit(Y);
time = toc;
[time]
Y = single(Y);
clear B;
clear cov;
clear U;
clear V;
clear eigenvalue; 
clear order;
clear R;
save isoh_48  Y W mvec;


%% test
% load isoh_48;
tep = find(Y<=0);
Y(tep) = -1;
clear tep;

tY = testdata*W-repmat(mvec,tn,1);
tY = (tY>0);
tB = compactbit(tY);
time = toc;
[time/tn]
clear tB;
tY = single(tY); 
tep = find(tY<=0);
tY(tep) = -1;
clear tep;

sim = Y*tY'; 
[temp,order] = sort(sim,1,'descend');
clear temp;
H = traingnd(order);
clear order;

ap = zeros(1,tn);
pre = zeros(1,tn);
interval = 500;
pt_num = 1+floor(n/interval);
prr = zeros(1,pt_num*2);
for i = 1:tn
    h = double(H(:,i) == testgnd(i));
    ind = find(h > 0);
    pn = length(ind);
    pre(i) = sum(h(1:range))/range;
%     if pn == 0
%         ap(i) = 0;
%     else
%         tep = 0;
%         for j = 1:pn
%             tep = tep+sum(h( 1:ind(j) ))/ind(j);
%         end
%         ap(i) = tep/pn;
%     end
%     clear ind;
   
    %% PR curve
    prr = prr+PR_new(h,interval);
    clear h;
end
prr = prr/tn;
[r, mean(pre,2), mean(ap,2)]

isoh_prr = prr;
plot(isoh_prr(pt_num+1:end),isoh_prr(1:pt_num),'b'); hold on; grid;

