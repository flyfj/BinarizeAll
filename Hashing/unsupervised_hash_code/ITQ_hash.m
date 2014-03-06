clear

addpath(genpath('../OWH/'));
% load cifar_split;
[traindata, traingnd, testdata, testgnd] = loadTrainingData(3);
[n,d] = size(traindata);
tn = size(testdata,1);
range = 100; 
r = 32;


%% ITQ
tic;
mvec = mean(traindata,1);
traindata = traindata-repmat(mvec,n,1);
cov = traindata'*traindata;
[U,V] = eig(cov);
eigenvalue = diag(V)';
[eigenvalue,order] = sort(eigenvalue,'descend');
W = U(:,order(1:r));
Y = traindata*W;

[temp, R] = ITQ(Y,50);
W = W*R;
mvec = mvec*W;

Y = Y*R;
Y = (Y>0);
B = compactbit(Y);
time = toc;
[time]
clear B;
clear cov;
clear U;
clear V;
clear eigenvalue;
clear order;
clear temp;
clear R;
Y = single(Y);
save itq_48  Y W mvec;

%% test
%load itq_48;
tep = find(Y<=0);
Y(tep) = -1;
clear tep;

tic;
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
[temp, order] = sort(sim,1,'descend');
clear temp;
H = traingnd(order);
clear order;

ap = zeros(1,tn);
pre = zeros(1,tn);
interval = 20;
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

itq_prr = prr;
plot(itq_prr(pt_num+1:end),itq_prr(1:pt_num),'b'); hold on; grid;

