function distMat = weightedHam( code1, code2, weights )
%WEIGHTEDHAM Summary of this function goes here
%   Detailed explanation goes here

distMat = zeros(size(code1, 1), size(code2, 1));

for i=1:size(code1, 1)
    for j=1:size(code2, 1)
        
        distMat(i,j) = ( double( code1(i,:)-code2(j,:) ) ).^2 * weights';
        
    end
end

