
load cifar_split;
[n,d] = size(traindata);
tn = size(testdata,1);
sig = 0.9876; 
m = 300;
t = 30;
range = 100;
r = 48;
load kernel_sample_300;


%% KLSH
tic;
Dis1 = sqdist(traindata(sample',:)',traindata(sample',:)'); 
Dis2 = sqdist(traindata',traindata(sample',:)');  
K1 = exp(-Dis1/(2*sig));
K2 = exp(-Dis2/(2*sig));
clear Dis1;
clear Dis2;

[H,W] = createHashTable(K1,r,t);
Y = ((K2*W)>0);
B = compactbit(Y);
time = toc;
[time]
clear B;
clear H;
clear K1;
clear K2;
Y = single(Y);
save klsh_48 Y W;


%% test
%load klsh_48;
tep = find(Y<=0);
Y(tep) = -1;
clear tep;

tic;
Dis3 = sqdist(testdata',traindata(sample',:)');
K3 = exp(-Dis3/(2*sig));
tY = ((K3*W)>0);
tB = compactbit(tY);
time = toc;
[time/tn]
clear tB;
clear Dis3;
clear K3;
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

klsh_prr = prr;
plot(klsh_prr(pt_num+1:end),klsh_prr(1:pt_num),'b'); hold on; grid;
