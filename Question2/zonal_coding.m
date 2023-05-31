function [CoeffMask,squared_error] = zonal_coding(B,p,q,n,percentage)
%Function to find mask for zonal coding Summary of this function goes here
%   B is block wise DCT of image
%   n is size of block
%p,q are image sizes
percentage_ofcoeffs_tokeep = percentage;
numberofcoeffs_tokeep = floor((n*n)*percentage_ofcoeffs_tokeep/100);
Store = zeros((p/n)*(q/n),n,n);
counter =1;
for i = 1:n:p
    for j=1:n:q
            Store(counter,:,:) = B(i:i+n-1,j:j+n-1);
            counter =counter+1;
    end
end
Mean_image_coeffs = zeros(n,n);
Var_image_coeffs = zeros(n,n);
for i = 1:n
    for j=1:n
            Mean_image_coeffs(i,j) =  mean(Store(:,i,j));
            Var_image_coeffs(i,j) = var(Store(:,i,j));
    end

end

sorted_block_variances = sort(Var_image_coeffs(:));
top_coeff = sorted_block_variances((length(sorted_block_variances)-numberofcoeffs_tokeep+1):length(sorted_block_variances));
threshold_value = top_coeff(1);
squared_error = sum(sorted_block_variances(1:(length(sorted_block_variances)-numberofcoeffs_tokeep)));
Var_image_coeffs((Var_image_coeffs)<threshold_value)=0;
Var_image_coeffs((Var_image_coeffs)>0)=1;%i am pretty sure it is right. Ask savvas
CoeffMask = Var_image_coeffs;
end