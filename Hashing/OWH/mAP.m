function mapval = mAP( p, r )
%MAP Summary of this function goes here
%   Detailed explanation goes here

mapval = 0;
for i=0:0.1:0.9
    pre = max( p(find(r>=i)) );
    mapval = mapval + pre;
end

mapval = mapval / 11;

end

