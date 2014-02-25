
load cifar_split;
[n,d] = size(traindata);
tn = size(testdata,1);
range = 100;
r = 48; % bit number


%% LSH 
tic;
W = randn(d,r); % generate hash functions
Y = traindata*W; 
mvec = mean(Y,1);   
Y = Y-repmat(mvec,n,1); % substract mean
t = max(abs(Y),[],1);
thres = rand(1,r).*t;   % generate threshold / bias
Y = Y+repmat(thres,n,1);
Y = (Y>0);
B = compactbit(Y);
time = toc;
[time]
clear B;
Y = single(Y);
save lsh_48_rand Y W mvec thres;


%% test
% load lsh_48_rand;  
tep = find(Y<=0);
Y(tep) = -1;
clear tep;

tic;
tY = testdata*W-repmat(mvec,tn,1)+repmat(thres,tn,1);
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

lsh_prr = prr;
plot(lsh_prr(pt_num+1:end),lsh_prr(1:pt_num),'b'); hold on; grid;

