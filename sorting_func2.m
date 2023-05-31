function [ret] = sorting_func2(block_struct)
%UNTITLED5 Only to compute mean sum of errors
%   Detailed explanation goes here
percent_of_coeffs_tokeep = 50;
numberofcoeffs_tokeep = floor(((8*8)*percent_of_coeffs_tokeep)/100);

sorted = sort(abs(block_struct.data(:)));
top_coeff = sorted(length(sorted)-numberofcoeffs_tokeep+1:length(sorted));%keeping top coeffs
threshold = top_coeff(1);%getting the lowest element as threshold
dummy = block_struct.data;
dummy(abs(block_struct.data)<threshold)=0;%zeroing all coeffs that are smaller than threshold  
dummy(abs(dummy)>0)=1;
ret = dummy;
end