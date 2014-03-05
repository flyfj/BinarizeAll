function prr = PR_new(H, k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% PR curve 
% Written by Wei Liu (wliu@ee.columbia.edu)
% H: ground truth vector in {0,1} 
% k: the regarded list length 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ln = sum(H);
n = size(H,1);
cn = floor(n/k);
pre = zeros(1,cn+1);
rec = pre;

% bug: handle no point
id = min(find( H(1:4*k)>0 ));
if isempty(id)
    pre(1) = 1 / (4*k);
else
    pre(1) = 1 / id;
end
rec(1) = 0;
for i = 1:cn
    range = i*k;
    if range > n
        range = n;
    end
    gn = sum(H(1:range)); 
    pre(i+1) = gn/range;
    rec(i+1) = gn/ln;
end
prr = [pre,rec];
