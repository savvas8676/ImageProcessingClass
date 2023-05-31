O = imread('flower.png');
A = im2double(O);%reading image
[m,n] = size(A);
%-----------------------------------------------------------------------------
%Noise addition
%First calculating noise variance 
m_noise = 0;%gaussian noise mean
P_signal = sum(A(:).^2)/(m*n);
var_gauss = P_signal/(10^(1.5)); %gaussian noise variance 

d = 0.25;%Percentage of pixels affected by Salt and Pepper Noise
Gaussian = imnoise(A,'gaussian',m_noise,var_gauss); %adds Gaussian white noise with mean m and variance var_gauss.
Salt_Pepper = imnoise(A,'salt & pepper',d); %adds salt and pepper noise, where d is the noise density. This affects approximately d*numel(I) pixels.

%Step 2 : Adding noise to the initial image both gaussian and salt and pepper separately 
A_saltpepper_noisy = Salt_Pepper; 
A_gaussian_noisy  = Gaussian;

%Preallocating arrays to store results of all size kernels for speed
B_median = zeros(9,m,n);
B_gaussian = zeros(9,m,n);
C = zeros(9,m,n);
C_median = zeros(9,m,n);
%Step 3 filtering with all types of kernels of size 3x3 to 11x11
for i=3:11
    B_median(i-2,:,:) = medfilt2(A_saltpepper_noisy,[i i]); %median filtering salt and pepper image with windows of size 3x3 to 11x11
    h = fspecial('average',i);
    B_gaussian(i-2,:,:) = medfilt2(A_gaussian_noisy,[i i]);%median filtering gaussian image with windows of size 3x3 to 11x11
    %---------------------------------------------------------------------------------------------------------------------------------
    C(i-2,:,:) = imfilter(A_gaussian_noisy,h,"same",'conv');%moving average filtering of gaussian image
    C_median(i-2,:,:) = imfilter(A_saltpepper_noisy,h,"same",'conv');%moving average filtering of salt and pepper image
end
%Displaying results 
%First display noisy images
figure;
subplot(3, 1, 1)
imshow(A,[]);
title("Initial Image")
subplot(3, 1, 2)
imshow(A_saltpepper_noisy,[]);
title("Initial Image with Salt and Pepper 25% percentage")
subplot(3, 1, 3)
imshow(A_gaussian_noisy,[]);
title("Initial Image with gaussian noise so as SNR is 15db")
print(gcf, '-dpng', 'Erotima3_png/erwtima3_1_NOISE.png');
%%
%Display results of median filtering salt and pepper noise in image
for i=3:11
    figure;
    temp = squeeze(B_median(i-2,:,:));
    imshow(temp,[])   
end
%%
%Display results of average filtering gaussian noise in image

for i=3:11
    figure;
    temp = squeeze(C(i-2,:,:));
    imshow(temp,[])   
end

%% 
%Display results of median filtering gaussian noise in image

for i=3:11
    figure;
    temp = squeeze(B_gaussian(i-2,:,:));
    imshow(temp,[])   
end

%% 
%Display results of average filtering salt and pepper noise in image

for i=3:11
    figure;
    temp = squeeze(C_median(i-2,:,:));
    imshow(temp,[])   
end


%%

figure;
subplot(3, 2, 1)
imshow(A,[]);
title("Initial Image")

subplot(3, 2, 2)
imshow(A_saltpepper_noisy,[]);
title("SandP 25% percentage")
subplot(3, 2, 3)
imshow(squeeze(B_median(4,:,:)),[]);
title("SandP best median filter 6x6")
subplot(3, 2, 4)
imshow(squeeze(C_median(5,:,:)),[]);
title("SandP best averaging filter 7x7")
subplot(3, 2, 5)
imshow(squeeze(B_median(1,:,:)),[]);
title("SandP  median filter 3x3")
print(gcf, '-dpng', 'Erotima3_png/erwtima3_2_NOISE.png');

figure;
subplot(2, 2, 1)
imshow(A,[]);
title("Initial Image")
subplot(2, 2, 2)
imshow(A_gaussian_noisy,[]);
title("Gaussian noise so as SNR is 15db")

subplot(2, 2, 3)
imshow(squeeze(B_gaussian(3,:,:)),[]);
title("Gaussian Noise best median filter 5x5")
subplot(2, 2, 4)
imshow(squeeze(C(4,:,:)),[]);
title("Gaussian Noise best filter 6x6")
print(gcf, '-dpng', 'Erotima3_png/erwtima3_3_NOISE.png');