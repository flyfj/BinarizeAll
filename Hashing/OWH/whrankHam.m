function distMat = whrankHam( code1_un, code1, code2, fstd )
%WHRANKDIST Summary of this function goes here
%   Detailed explanation goes here


num2 = size(code2, 1);

distMat = zeros(size(code1, 1), size(code2, 1));

for i=1:size(code1, 1)
    
    % compute query-dependent weight
    weights = abs(code1_un(i,:)) ./ fstd;
    
    codediff = abs(repmat(code1(i,:), num2, 1) - code2);
%     if type == 1
%         codediff = double(2*codediff - 1);
%     end
    distMat(i,:) = weights * codediff';
    
end


end

