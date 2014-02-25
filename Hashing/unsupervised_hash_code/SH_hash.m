
load cifar_split;
[n,d] = size(traindata);
tn = size(testdata,1);
range = 100;
r = 48;


%% SH 
tic
SHparam.nbits = r;
SHparam = trainSH(traindata, SHparam);
[B,Y] = compressSH(traindata, SHparam);
time = toc;
[time]
clear B;
save sh_48 Y SHparam;


%% test
% load sh_48;
tep = find(Y<=0);
Y(tep) = -1;
clear tep;

tic;
[tB,tY] = compressSH(testdata, SHparam);
time = toc;
[time/tn]
clear tB;
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

sh_prr = prr;
plot(sh_prr(pt_num+1:end),sh_prr(1:pt_num),'b'); hold on; grid;

