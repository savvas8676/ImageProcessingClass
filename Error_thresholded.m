function [ems_] = Error_thresholded(Masks,B,p,q,n)
%Function to find mask for zonal coding Summary of this function goes here
%   B DCT coeffs
%   Masks is block wise masks of DCT of image
%   n is size of block
%p,q are image sizes

Store = zeros((p/n)*(q/n),n,n);
counter = 1;
for i = 1:n:p
    for j=1:n:q
            Store(counter,:,:) = B(i:i+n-1,j:j+n-1);
        counter = counter+1;
    end
    
end

Var_image_coeffs = zeros(n,n);
for i = 1:n
    for j=1:n
            
            Var_image_coeffs(i,j) = var(Store(:,i,j));
    end

end
ems = zeros(1,(p/n)*(q/n));
counter = 1;
for i = 1:n:p 
    for j=1:n:q
            A = Masks(i:i+n-1,j:j+n-1);
            ems(counter) = sum(Var_image_coeffs(A==0));%Computing the ems error
        counter =counter+1;
    end
end
ems_ = sum(ems)/length(ems);%averaging ems of all subimages


end