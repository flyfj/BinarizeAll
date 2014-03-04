function distMat = svmdist( code1, code2, svmmodel, type )
%SVMDIST Summary of this function goes here
%   Detailed explanation goes here


num2 = size(code2, 1);

distMat = zeros(size(code1, 1), size(code2, 1));

for i=1:size(code1, 1)
    
    codediff = abs(repmat(code1(i,:), num2, 1) - code2);
    if type == 1
        codediff = double(2*codediff - 1);
    end
    
    % do svm prediction
    [~, ~, scores] = svmpredict(ones(size(codediff,1), 1), codediff, svmmodel);
    
    distMat(i,:) = scores;
    
end


end

