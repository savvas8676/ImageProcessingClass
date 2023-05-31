clear;
clc;
O = rgb2gray(imread('factory.jpg'));
I = O;%reading image
[m,n] = size(I);
%-----------------------------------------------------------------------------
figure;
subplot(5,4,1)
imshow(I,[])
title("Initial image")
subplot(5,4,2)
imshow(20*log(1+abs(fftshift(fft2(I)))),[])
title("Initial image fourier transform centered")
Fourier_Image = fftshift(fft2(I));
%Plotting impulse response of degradation system
sigma =4;
h = fspecial("gaussian",[m,n],sigma);
subplot(5,4,3)
plot(h(m/2,:))
title("Impulse response line horizontal at the center")
subplot(5,4,4)
plot(h(:,n/2))
title("Impulse response line vertical at the center")
subplot(5,4,5)
imshow(log(1+h),[])
title("Impulse response as an image")

subplot(5,4,6)
imshow(20*log(1+abs(fftshift(fft2(h)))),[])
title("Gaussian fourier transform centered")
H = fftshift(fft2(h));

%Linear degradation of initial image in the frequency domain
Filtered_Image_Fourier = Fourier_Image.*H;
subplot(5,4,7)
imshow(20*log(1+abs(Filtered_Image_Fourier)),[])
title("Initial image FT degraded")
subplot(5,4,8)
Degraded_image = fftshift(ifft2(fftshift(Filtered_Image_Fourier)));
imshow(Degraded_image ,[])
title("Degraded Image")

%Adding nosise to degraded image of 10DB 
P_signal = sum(I(:).^2)/(m*n);
noise_var = 3*(P_signal/(10^(1)));
Noisy_degraded_image = Degraded_image + sqrt(noise_var)*randn(m,n);
subplot(5,4,9)
imshow(Noisy_degraded_image ,[])
title("Degraded-Noisy Image") 

Noisy_degraded_image_FT = fftshift(fft2(Noisy_degraded_image));
subplot(5,4,10)
imshow(log(1+abs(Noisy_degraded_image_FT)) ,[])
title("Degraded-Noisy Image FT") 


%Wiener filtering 
P_G = (abs(Noisy_degraded_image_FT).^2)/m*n;
P_F_with_knowledge = P_G - ones(m,n)*noise_var;
%Estimating Noise PSD
n_win =50;%Window of estimation in high frequencies where signal content can be considered negligible
Estimated_noise_var = mean2(Noisy_degraded_image_FT(m-n_win+1:m,n-n_win+1:n));
P_F_without_knowledge = P_G - ones(m,n)*Estimated_noise_var;
%Constructing wiener filters
Wiener1 = P_F_with_knowledge ./(P_F_with_knowledge + ones(m,n)*noise_var);
Wiener2 = P_F_with_knowledge ./(P_F_with_knowledge + ones(m,n)*Estimated_noise_var);
%Denoising with above filters
Denoised_I_wiener_1 = Noisy_degraded_image_FT.*Wiener1;
Denoised_I_wiener_2 = Noisy_degraded_image_FT.*Wiener2;
%Plotting Results
Degraded_image_denoised_1 = (ifft2(fftshift(Denoised_I_wiener_1)));
subplot(5,4,11)
imshow(Degraded_image_denoised_1 ,[])
title("Degraded Image after wnr with knowledge")
Degraded_image_denoised_2 = (ifft2(fftshift(Denoised_I_wiener_2)));
subplot(5,4,12)
imshow(Degraded_image_denoised_2 ,[])
title("Degraded Image after wnr without knowledge")

FT_noise = Noisy_degraded_image_FT - Filtered_Image_Fourier;
subplot(5,4,13)
imshow(log(1+(abs(FT_noise))) ,[])
title("FT of Noise image")
%Just estimation of noise variance in a flat area of the noisy image
temp = Noisy_degraded_image(370:380,600:610);
VAR_SPACE = var(temp(:));
%----------------------------------------------------------------------------



%Inverse filtering 
%Step 1 :Threshold frequency response so as to prevent Inf values
th = 1;
Temp_H = H;
H_inv_no_threshold = ones(m,n)./Temp_H;
Temp_H(abs(H)<th)= th;
H_inv = ones(m,n)./Temp_H;

%Inverse filter
inv_filter_1  = Denoised_I_wiener_1.*H_inv;

inv_filter_2  = Denoised_I_wiener_2.*H_inv;
%Inverse filter with no threshold
inv_filter_1_nothreshold  = Denoised_I_wiener_1.*H_inv_no_threshold;

inv_filter_2_nothreshold  = Denoised_I_wiener_2.*H_inv_no_threshold;
%Plotting Results
Restored_image_denoised_1 = (ifft2(fftshift(inv_filter_1)));
subplot(5,4,14)
imshow(Restored_image_denoised_1 ,[])
title("Restored after wnr knowledge + inv")
Restored_image_denoised_2 = (ifft2(fftshift(inv_filter_2)));
subplot(5,4,15)
imshow(Restored_image_denoised_2 ,[])
title("Restored after wnr no-knowledge + inv")


%Plotting Results

% figure;
% Restored_image_denoised_1_nothreshold = (ifft2(fftshift(inv_filter_1_nothreshold)));
% imshow(Restored_image_denoised_1_nothreshold ,[]);
% print(gcf, '-dpng', 'Erotima5_png/Inverse_no_threshold_knowledge.png');
% title("Restored after wnr knowledge + inv without thresold")
% figure;
% Restored_image_denoised_2_nothreshold = (ifft2(fftshift(inv_filter_1_nothreshold)));
% imshow(Restored_image_denoised_2_nothreshold ,[]);
% title("Restored after wnr no-knowledge + inv without thresold")
% print(gcf, '-dpng', 'Erotima5_png/Inverse_no_threshold_no_knowledge.png');

%Algorithm 2:Wiener deconvolution
H_conj = conj(H);
power_spectrum = abs(H).^2;
Wienerdeconv_1 = (power_spectrum.*P_F_with_knowledge)./(P_F_with_knowledge.*power_spectrum + (noise_var*ones(m,n))).*H_inv; 
Wienerdeconv_2 =  (power_spectrum.*P_F_without_knowledge)./(P_F_without_knowledge.*power_spectrum + (noise_var*ones(m,n))).*H_inv; 
%Filtering
deconv_1 = Noisy_degraded_image_FT.*Wienerdeconv_1;
deconv_2 = Noisy_degraded_image_FT.*Wienerdeconv_2;

%Plotting Results
Restored_image_deconv_1 = (ifft2(fftshift(deconv_1)));
subplot(5,4,16)
imshow(Restored_image_deconv_1 ,[])
title("Restored after deconv_1")
Restored_image_deconv_2 = (ifft2(fftshift(deconv_2)));
subplot(5,4,17)
imshow(Restored_image_deconv_2 ,[])
title("Restored after deconv_2")

print(gcf, '-dpng', 'Erotima5_png/erwtima5_1_Initial.png');