%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm for filtering in the frequency Domain
% Step1: Given an input image f(x,y) of size M x N, obtain the pading
% parameters P and Q. Typically, we select P = 2M and Q = 2N
% Step2: Form a padded image fp(x,y) of size P X Q by appending the
% necessary number of zeros to f(x,y).
% Step3: Multiply fp(x,y) by (-1)^(x+y) to center its transform
% Step4: Compute the DFT, F(u,v) of the image from Step 3
% Step5: Generate a Real, Symmetric Filter Function H(u,v) of size P X Q
% with center at coordinates (P/2,Q/2),
% Step 6:Form the product G(u,v) = H(u,v)F(u,v) using array multiplication
% Obtain the processed image
% Step 7: gp(x,y) = {real{inverse DFT[G(u,v)]}(-1)^(x+y)
% Step 8: Obtain the final processed result g(x,y) by extracting the M X N region
% from the top, left quadrant of gp(x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm Implementation
% Step1: Given an input image f(x,y) of size M x N, obtain the pading
% parameters P and Q. Typically, we select P = 2M and Q = 2N
aerial = imread('aerial.tiff');
% Converting the image class into "double"
aerial = im2double(aerial);%255*
% reading the image size
[m,n] = size(aerial);
% Step 2
% creating padded aerial image of size 2m X 2n
%configuring required size of the padded image
p = 2*m;
q=2*n;
padded_aerial = [aerial zeros(m,q-n);zeros(p-m,q)];
%showing the output of the padding of the image, before and after
figure;
imshow(uint8(aerial));title('original version of the  image');
figure;
imshow(uint8(padded_aerial));title('padded version of the  image');
%-------------------------------------------------------
% Step 3
% creating a null array of size p X q
shifted_padded_aerial = zeros(p,q);
% Multiplying the padded image with (-1)^(x+y)
for i = 1:p
    for j = 1:q
        shifted_padded_aerial(i,j) = padded_aerial(i,j).*(-1).^(i + j);
    end
end
figure;
imshow(uint8(shifted_padded_aerial));title('Padded image with centered spectrum');

% Step 4
% Computing the 2D DFT of the original image using "fft2" matlab command
%As a preporcessing measure we center it
shifted_aerial = zeros(m,n);
for i = 1:m
    for j = 1:n
        shifted_aerial(i,j) = aerial(i,j).*(-1).^(i + j);
    end
end
Dft_shifted_padded_aerial = fft2(shifted_padded_aerial);
%-------------------------------------------------------------
%ploting different spectrums for centered spectrum image, uncenrered
%original image and padded centered spectrum image
%unpadded image with shifting, and calculating fourier spectrum
centerd_original_image_dft = fft2(shifted_aerial);
centerd_original_image_dft_abs = abs(centerd_original_image_dft);
log_scale_centerd_original_image_dft = log(1+centerd_original_image_dft_abs);
figure(1);
imshow(centerd_original_image_dft_abs,[]);
title('2D DFT of the  image abs with shifting');
figure(2);
imshow(log_scale_centerd_original_image_dft,[]);
title('2D DFT of the  image log of 1+abs with shifting');
%unpadded original image without shifting and padding-----------------------------------
original_image_dft = fft2(aerial);
original_image_dft_abs = abs(original_image_dft);
log_scale_original_image_dft = log(1+original_image_dft_abs);
figure(3);
imshow(original_image_dft_abs,[]);
title('2D DFT of the original image abs without shifting ');
figure(4);
imshow(log_scale_original_image_dft,[]);
title('2D DFT of the original image log of 1+abs without shifting');
%padded and shifted image spectrum------------------------------------------------------
Dft_shifted_padded_aerial_abs = abs(Dft_shifted_padded_aerial);
log_scale_centerd_padded_image_dft = log(1+Dft_shifted_padded_aerial_abs);
figure(5);
imshow(Dft_shifted_padded_aerial_abs,[]);
title('2D DFT of the  image abs with shifting and padding');

figure(6);
imshow(log_scale_centerd_padded_image_dft,[]);
title('2D DFT of the image log of 1+abs with shifting and padding');

%%
%%%%%%%%%%%%%%%%%%%%
% Step 5
% Generating the Real, Symmetric Filter Function ideal low pass and
% high_pass filters_ cuttof is epressed in pixel distance from center as
% per the book
%
% command
%d_zero=102;
d_zero=0.2;
[X,Y] = freqspace([p,q],"meshgrid");
z = zeros(p,q);
% for i = 1:p
%     for j = 1:q
%         z(i,j) = sqrt((i-p/2).^2 + (j-q/2).^2);
%     end
% end
for i = 1:p
    for j = 1:q
        z(i,j) = sqrt((X(i,j)).^2 + (Y(i,j)).^2);
    end
end
% Choosing the Cut off Frequency and hence defining the ideal low pass filter
% mask
H = zeros(p,q);
for i = 1:p
    for j = 1:q
        if z(i,j) <= d_zero  % here 0.4 is the cut-off frequency of the LPF
            H(i,j) = 1;
        else
            H(i,j) = 0;
        end
    end
end
H_lowpass=H;
H_highpass=1-H;
%----------------------------------------------------------------
%shifting , in order to get properly centered spatial
%representation
lowpass_filter_impulse_response_notshifted=real(fftshift(ifft2(H_lowpass)));
highpass_filter_impulse_response_notshifted=real(fftshift(ifft2(H_highpass)));
lowpass_filter_impulse_response = zeros(size(H_lowpass));
for i = 1:p
    for j = 1:q
        lowpass_filter_impulse_response(i,j) = lowpass_filter_impulse_response_notshifted(i,j).*(-1).^(i + j);
    end
end
highpass_filter_impulse_response = zeros(size(H_highpass));
for i = 1:p
    for j = 1:q
        highpass_filter_impulse_response(i,j) = highpass_filter_impulse_response_notshifted(i,j).*(-1).^(i + j);
    end
end

%-----------------------------------
%showing frequency responses of both ideal filters created
figure;
imshow(H_lowpass,[]);
title('Low Pass Filter Mask, in the frequence domain');
print(gcf, '-dpng', 'Erotima1_png/erwtima1_2_Lowpass_frequency.png');
figure;
imshow(H_highpass,[]);
title('High Pass Filter Mask, in the frequence domain');
print(gcf, '-dpng', 'Erotima1_png/erwtima1_2_Highpass_frequency.png');
%-----------------------------------
%showing frequency responses of both ideal filters created
figure;


figure;
imshow(lowpass_filter_impulse_response,[]);
title('Low Pass Filter Mask, in the spatial domain(impulse response)');
print(gcf, '-dpng', 'Erotima1_png/erwtima1_2_Lowpass_impulsefull.png');
figure;
imshow(highpass_filter_impulse_response,[]);
title('High Pass Filter Mask, in the spatial domain (impulse response)');
print(gcf, '-dpng', 'Erotima1_png/erwtima1_2_Highpass_impulsefull.png');
figure;
plot(highpass_filter_impulse_response(257,:));
title("High pass filter 1-D impulse response")
print(gcf, '-dpng', 'Erotima1_png/erwtima1_2_Highpass_impulse1d.png');

plot(lowpass_filter_impulse_response(256,:));
title("Lowpass filter 1-D impulse response")
print(gcf, '-dpng', 'Erotima1_png/erwtima1_2_lowpass_impulse1d.png');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 6:Form the product G(u,v) = H(u,v)F(u,v) using array multiplication
% Obtain the processed image
% from the previous program lines we know that,
% Dft_shifted_padded_aerial : the 2D DFT output of pre processed image
% H : the mask for Low Pass Filter
% let final_blurred_image is the variable of the final result
low_pass_filtered_image_frequency_domain = Dft_shifted_padded_aerial.*H_lowpass;
high_pass_filtered_image_frequency_domain = Dft_shifted_padded_aerial.*H_highpass;
%low_pass_filtered_image_frequency_domain = zeros(p,q);
%low_pass_filtered_image_frequency_domain(256,256) = Dft_shifted_padded_aerial(256,256);
low_pass_filtered_image_frequency_domain_log_scale = log(1+(abs(low_pass_filtered_image_frequency_domain)));
high_pass_filtered_image_frequency_domain_log_scale = log(1+(abs(high_pass_filtered_image_frequency_domain)));
figure;
imshow(abs(low_pass_filtered_image_frequency_domain_log_scale),[]);title('Low pass filtering with ideal filter frequency output image afterwards');
figure;
imshow(abs(high_pass_filtered_image_frequency_domain_log_scale),[]);title('High pass filtering with ideal filter frequency output image afterwards');
%----------------------------------------------------------------------------------
% Step 7: gp(x,y) = {real{inverse DFT[G(u,v)]}(-1)^(x+y)
% calculation of inverse 2D DFT of the "out"
low_pass_filtered_image_spatial_domain_centered_spectrum = real(ifft2(low_pass_filtered_image_frequency_domain));
high_pass_filtered_image_spatial_domain_centered_spectrum = real(ifft2(high_pass_filtered_image_frequency_domain));
figure(15);
imshow(low_pass_filtered_image_spatial_domain_centered_spectrum,[]);title('output image after inverse 2D DFT');
figure;
imshow(high_pass_filtered_image_spatial_domain_centered_spectrum,[]);title('output image after inverse 2D DFT');
% post process operation of getting the original iamge, in proportion
low_pass_filtered_image_spatial_domain_non_centered_spectrum = zeros(p,q);
for i = 1:p
    for j = 1:q
        low_pass_filtered_image_spatial_domain_non_centered_spectrum(i,j) = low_pass_filtered_image_spatial_domain_centered_spectrum(i,j).*((-1).^(i+j));
    end
end
figure;
imshow(low_pass_filtered_image_spatial_domain_non_centered_spectrum,[]);title('Post Processed low pass filtered image');
high_pass_filtered_image_spatial_domain_non_centered_spectrum = zeros(p,q);
for i = 1:p
    for j = 1:q
        high_pass_filtered_image_spatial_domain_non_centered_spectrum(i,j) = high_pass_filtered_image_spatial_domain_centered_spectrum(i,j).*((-1).^(i+j));
    end
end
figure;
imshow(high_pass_filtered_image_spatial_domain_non_centered_spectrum,[]);title('Post Processed high pass filtered image');
%--------------------------------------------------------------------------------------------
% Step 8: Obtain the final processed result g(x,y) by extracting the M X N region
% from the top, left quadrant of gp(x,y)
% let the smoothed image or low pass filtered image is "out"
final_blurred_image = zeros(m,n);
for i = 1:m
    for j = 1:n
        final_blurred_image(i,j) = low_pass_filtered_image_spatial_domain_non_centered_spectrum(i,j);
    end
end
figure;
%imshow([uint8(aerial) uint8(final_blurred_image)],[]);title('input image                 output image');
subplot(1,2,1)
imshow(aerial,[])
title('input image');
subplot(1,2,2)
imshow(final_blurred_image,[])
title('filtered lowpass image 40% of frequencies kept');
print(gcf, '-dpng', 'Erotima1_png/erwtima1_4_imageslowpass.png');
%-------------------------------------------------------
%highpass filtering result
final_sharpened_image = zeros(m,n);
for i = 1:m
    for j = 1:n
        final_sharpened_image(i,j) = high_pass_filtered_image_spatial_domain_non_centered_spectrum(i,j);
    end
end
%final_sharpened_image(final_sharpened_image>0.5*max(final_sharpened_image(:))) = 0.5*max(final_sharpened_image(:));

%imshow([uint8(aerial) uint8(final_blurred_image)],[]);title('input image                 output image');
subplot(1,2,1)
imshow(aerial,[])
title('input image');
subplot(1,2,2)
imshow(final_sharpened_image)
title('filtered Highpass image 10% of frequencies dropped');     
print(gcf, '-dpng', 'Erotima1_png/erwtima1_4_imageshighpass.png');