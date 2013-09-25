function [ codes ] = compress2Base( samps, hash_params, type )
%COMPRESS2BASE Summary of this function goes here
%   Detailed explanation goes here

if( strcmp(type, 'LSH') )
    
    codes = int8( samps * hash_params.funcs' > 0 );
    
end


end

