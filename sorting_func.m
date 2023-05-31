function [ret] = sorting_func(block_struct,percentage)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
percent_of_coeffs_tokeep = percentage;
numberofcoeffs_tokeep = floor(((32*32)*percent_of_coeffs_tokeep)/100);

sorted = sort(abs(block_struct.data(:)));
top_coeff = sorted(length(sorted)-numberofcoeffs_tokeep+1:length(sorted));%keeping top coeffs
threshold = top_coeff(1);%getting the lowest element as threshold

block_struct.data(abs(block_struct.data)<threshold)=0;%zeroing all coeffs that are smaller than threshold
ret = block_struct.data;
end








