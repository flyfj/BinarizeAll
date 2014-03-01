function distMat = weightedHam( code1, code2, weights, type )
%WEIGHTEDHAM Summary of this function goes here
%   weights: row vector
%   type: 0 - 0/1; 1 - -1/1

% num1 = size(code1, 1);
num2 = size(code2, 1);

distMat = zeros(size(code1, 1), size(code2, 1));

for i=1:size(code1, 1)
    
    codediff = abs(repmat(code1(i,:), num2, 1) - code2);
%     codediff = double(2*codediff - 1);
%     if type == 1
%         codediff = double(2*codediff - 1);
%     end
    distMat(i,:) = weights * codediff';
    
end

