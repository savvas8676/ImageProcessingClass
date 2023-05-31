function [ems] = Error_thresholded_2(G,G_comp,p,q,n)
%calculating mean squared error of compression using a different way

ems = zeros(1,(p/n)*(q/n));
counter = 1;
for i = 1:n:p
    for j=1:n:q
       
            ems(counter) = (norm(G(i:i+n-1,j:j+n-1)-G_comp(i:i+n-1,j:j+n-1))).^2;%Computing the ems error
 
        counter =counter+1;
    end
end

ems = mean(ems);

end