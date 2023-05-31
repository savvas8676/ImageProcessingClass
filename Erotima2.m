%script file for question 2
%clearing console
clc;
clear;
%reading the image in 32-bit form
ems_error1 = zeros(1,46);
lenna_image = rgb2gray(imread('lenna.jpg'));
lenna_dimensions=size(lenna_image);
lenna_image = im2double(lenna_image);
lenna_image = [lenna_image zeros(257,1); zeros(288-lenna_dimensions(1),256)];
figure;
imshow(lenna_image,[]);%initial image
title("Lenna padded image");
for percentageofcoeffs  = 5:50
T = dctmtx(32);%creating DCT matrix of size 32x32
dct = @(block_struct) T * block_struct.data * T';
thresholded_ = @(block_struct) sorting_func(block_struct,percentageofcoeffs);

lenna_block_dct = blockproc(lenna_image,[32 32],dct,"UseParallel",true);%perform DCT in each block and store the result back to the matrix
%-------------------------------------------------------------
%thresholding method on dct block magnitude
lenna_block_dct_compressed = blockproc(lenna_block_dct,[32 32],thresholded_,"UseParallel",true);%mask the DCT coeffs based on the mask result stored in B2


%reconstructing the image with the remaining coeffs in each block
invdct = @(block_struct) T' * block_struct.data * T;
lenna_compressed = blockproc(lenna_block_dct_compressed,[32 32],invdct);%performing idct in each block
%------------------------------------------------

%computing and plotting mean squared error between original and compressed
%image
ems_error1(percentageofcoeffs-4)=sqrt(mean2((255*lenna_image-255*lenna_compressed).^2));
if percentageofcoeffs == 5
    figure;
    imshow([lenna_image(1:lenna_dimensions(1),1:lenna_dimensions(2)) lenna_compressed(1:lenna_dimensions(1),1:lenna_dimensions(2))],[])
    title("Left is initial image right is compressed image using 5 percent of coeffs")
    print(gcf, '-dpng', 'Erotima2_png/erwtima1image5.png');
end
end
% %Showing results
% lenna_compressed = lenna_compressed(1:lenna_dimensions(1),1:lenna_dimensions(2));
% figure;
% imshow(lenna_compressed,[])
%---------------------------------------------------
%computing and plotting mean squared error between original and compressed
%thresholding version
%image

% %Showing results
lenna_compressed = lenna_compressed(1:lenna_dimensions(1),1:lenna_dimensions(2));
figure;
imshow([lenna_image(1:lenna_dimensions(1),1:lenna_dimensions(2)) lenna_compressed],[])
title("Left is initial image right is compressed image using 50 percent of coeffs")
print(gcf, '-dpng', 'Erotima2_png/erwtima1image50.png');
figure;
x=5:50;
plot(x,ems_error1,Marker = "+",MarkerFaceColor="white",MarkerSize=8,Color="red",LineWidth=2)
xlabel('percentage of information kept') 
ylabel('Mean squared error') 
title("Thresholded coding error plot for percentage 5:50")
print(gcf, '-dpng', 'Erotima2_png/erwtima1ploterrorthresholded.png');


%script file for question 2 which is zonal coding 
%clearing console

%reading the image in 32-bit form
%percentageofcoeffs = 50;
ems_error2 = zeros(1,46);
lenna_image = rgb2gray(imread('lenna.jpg'));
lenna_dimensions=size(lenna_image);
lenna_image = im2double(lenna_image);
lenna_image = [lenna_image zeros(257,1); zeros(288-lenna_dimensions(1),256)];
lenna_dimensions2=size(lenna_image);
imshow(lenna_image,[]);%initial image
for percentageofcoeffs  = 5:50
counter=0;

T = dctmtx(32);%creating DCT matrix of size 32x32
dct = @(block_struct) T * block_struct.data * T';

lenna_block_dct = blockproc(lenna_image,[32 32],dct,"UseParallel",true);%perform DCT in each block and store the result back to the matrix
%-------------------------------------------------------------
%zonal coding method on dct block magnitude
[block_mask ,squared_error] =  zonal_coding(lenna_block_dct,lenna_dimensions2(1),lenna_dimensions2(2),32,percentageofcoeffs);
thresholded_ = @(block_struct) block_struct.data .*block_mask;
lenna_block_dct_compressed = blockproc(lenna_block_dct,[32 32],thresholded_,"UseParallel",true);%mask the DCT coeffs based on the mask result stored in B2


%reconstructing the image with the remaining coeffs in each block
invdct = @(block_struct) T' * block_struct.data * T;
lenna_compressed = blockproc(lenna_block_dct_compressed,[32 32],invdct);%performing idct in each block
%---------------------------------------------------
%computing and plotting mean squared error between original and compressed
%image
ems_error2(percentageofcoeffs-4)=sqrt(mean2((255*lenna_image-255*lenna_compressed).^2));
%------------------------------------------------
if percentageofcoeffs == 5
    figure;
    imshow([lenna_image(1:lenna_dimensions(1),1:lenna_dimensions(2)) lenna_compressed(1:lenna_dimensions(1),1:lenna_dimensions(2))],[])
    title("Left is initial image right is compressed image using 5 percent of coeffs")
    print(gcf, '-dpng', 'Erotima2_png/erwtimaimage5.png');
end

end
%Showing results
lenna_compressed = lenna_compressed(1:lenna_dimensions(1),1:lenna_dimensions(2));
figure;
imshow([lenna_image(1:lenna_dimensions(1),1:lenna_dimensions(2)) lenna_compressed],[])
title("Left is initial image right is compressed image using 50 percent of coeffs")
print(gcf, '-dpng', 'Erotima2_png/erwtimaimage50.png');
figure;
x=5:50;
plot(x,ems_error2,Marker = "+",MarkerFaceColor="white",MarkerSize=8,Color="red",LineWidth=2)
xlabel('percentage of information kept') 
ylabel('Mean squared error') 
title("Zonal coding error plot for percentage 5:50")
print(gcf, '-dpng', 'Erotima2_png/erwtimaploterrorzonal.png');

%%
figure;
x=5:50;

plot(x,ems_error2,"Color","green","Marker","^");
hold on;
plot(x,ems_error1,"Marker","+","Color","cyan")
xlabel('percentage of information kept') 

ylabel('Mean squared error ') 

title("Zonal coding error vs Thresholded plot for percentage 5:50")
print(gcf, '-dpng', 'Erotima2_png/erwtimaploterrorzonalvsthresholded.png');
legend('Zonal Coding','Thresholded Coding','Location','northeast')
