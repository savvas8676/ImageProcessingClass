a_1 = imread('dark_road_1.jpg');
a_2 = imread("dark_road_2.jpg");
a_3 = imread("dark_road_3.jpg");

% Converting the image class into "double"
b_1 = im2double(a_1);
b_2 = im2double(a_2);
b_3 = im2double(a_3);

% Converting the double  into "uint8"
% b_1 = uint8(255*b_1);
% b_2 = uint8(255*b_2);
% b_3 = uint8(255*b_3);
% reading the image size
[M_1,N_1] = size(b_1);
[M_2,N_2] = size(b_2);
[M_3,N_3] = size(b_3);
figure;
imshow(b_1,[]);title('original image dark road 1');

figure;
imshow(b_2,[]);title('original image dark road 2');


figure;
imshow(b_3,[]);title('original image dark road 3');
%%
%coputing histogram of all images and plotting it
figure;
subplot(2,3,1)
imhist(b_1,256)
subplot(2,3,2)
imhist(b_2,256)
subplot(2,3,3)
[counts,binLocations] = imhist(b_3,256);
stem(binLocations,counts,"Marker",".")
%%
figure(1);
subplot(2, 3, 1), imshow(log(1+255*b_1),[]);title('original image dark road 1');
subplot(2, 3, 2), imshow(log(1+255*b_2),[]);title('original image dark road 2');
subplot(2, 3, 3), imshow(log(1+255*b_3),[]);title('original image dark road 3');
subplot(2,3,4)
[counts,binLocations] = imhist(uint8(log(1+255*b_1)),256);
stem(binLocations,counts,"Marker",".")
subplot(2,3,5)
[counts,binLocations] = imhist(uint8(log(1+255*b_2)),256);
stem(binLocations,counts,"Marker",".")
subplot(2,3,6)
[counts,binLocations] = imhist(uint8(log(1+255*b_3)),256);
stem(binLocations,counts,"Marker",".")
print(gcf, '-dpng', 'Erotima4_png/erwtima4_2_Histograms.png');
%%
%Equalizing histogram using global histogram equalization
number_of_output_bins = 255;
g_enhanced_1 = histeq(b_1,255);
g_enhanced_2 = histeq(b_2,255);
g_enhanced_3 = histeq(b_3,255);
subplot(2,3,4)
imhist(g_enhanced_1);
subplot(2,3,5)
imhist(g_enhanced_2);
subplot(2,3,6)
imhist(g_enhanced_3);

%%
figure(1);
subplot(2, 3, 1), imshow(255*g_enhanced_1,[]);title('original image dark road 1');
subplot(2, 3, 2), imshow(255*g_enhanced_2,[]);title('original image dark road 2');
subplot(2, 3, 3), imshow(255*g_enhanced_3,[]);title('original image dark road 3');
subplot(2,3,4)
[counts,binLocations] = imhist(uint8(255*g_enhanced_1),256);
stem(binLocations,counts,"Marker",".")
subplot(2,3,5)
[counts,binLocations] = imhist(uint8(255*g_enhanced_2),256);
stem(binLocations,counts,"Marker",".")
subplot(2,3,6)
[counts,binLocations] = imhist(uint8(255*g_enhanced_3),256);
stem(binLocations,counts,"Marker",".")
print(gcf, '-dpng', 'Erotima4_png/erwtima4_enhanced_Histograms.png');
%%
%showing images after global histogram equalization
figure;
imshow(g_enhanced_1);title('original image dark_road_1 globally equalized');
figure;
imhist(b_1)
figure;
imhist(im2double(g_enhanced_1));

%%

figure;
imshow(g_enhanced_2);title('original image dark_road_2 globally equalized');


figure;
imshow(g_enhanced_3);title('original image dark_road_3  globally equalized');
%%
%local histogram equalization
%b_2 = imread('tire.tif');
%b_2 = im2double(b_2);
I_1 = uint8(255*im2double(b_1));
I_2 = uint8(255*im2double(b_2));
I_3 = uint8(255*im2double(b_3));

neighborhood = 180;
%fun = @(x) local(x);
%B_1 = nlfilter(b_1,[neighborhood neighborhood],fun);
[M1,N1] = size(I_1);
[M2,N2] = size(I_2);
[M3,N3] = size(I_3);
step  = floor(neighborhood/2);
p = M+(2*step);
q = N+(2*step);
L=256;%number of intensity levels
% b_2_padded  = uint8(zeros(p,q));%allocating space for padded image
% icounter=1;
% jcounter=1;
% for i = step+1:p-step
% 
%     for j = step+1:q-step
%         b_2_padded(i,j) = b_2(icounter,jcounter);
%         jcounter=jcounter+1;
%     end
%     icounter=icounter+1;
%     jcounter=1;
% end
%b_2_padded = padarray(b_2,[step step],0,'both');
%b_3_padded = padarray(b_2,[step step],'replicate','both');
%Finished padding the image 
I_1_dummy = zeros(M1,N1);
I_2_dummy = zeros(M2,N2);
I_3_dummy = zeros(M3,N3);
%b_3_padded_dummy = zeros(p,q);
%Locally equalizing each pixel
for i = 1:M1%step+1:p-step
    for j = 1:N1%step+1:q-step
        I_1_dummy(i,j) = round((L-1)*local_hist_kernel(I_1,i,j,neighborhood));
        
        %b_3_padded_dummy(i,j) = round((L-1)*local_hist_kernel(b_3_padded,i,j,neighborhood));
    end
end

        
for i = 1:M2%step+1:p-step
    for j = 1:N2%step+1:q-step
        I_2_dummy(i,j) = round((L-1)*local_hist_kernel(I_2,i,j,neighborhood));
        
        %b_3_padded_dummy(i,j) = round((L-1)*local_hist_kernel(b_3_padded,i,j,neighborhood));
    end
end
for i = 1:M3%step+1:p-step
    for j = 1:N3%step+1:q-step
        I_3_dummy(i,j) = round((L-1)*local_hist_kernel(I_3,i,j,neighborhood));
        
        %b_3_padded_dummy(i,j) = round((L-1)*local_hist_kernel(b_3_padded,i,j,neighborhood));
    end
end
% B_2 = b_2_padded_dummy(step+1:p-step,step+1:q-step);%dropping all zeros from the padded image
% B_3 = b_3_padded_dummy(step+1:p-step,step+1:q-step);

figure;
subplot(2,3,1)
imshow(I_1_dummy,[]);title('Ιmage Dark road 1 locally equalized');
subplot(2,3,2)
imshow(I_2_dummy,[]);title('Ιmage Dark road 2 locally equalized');
subplot(2,3,3)
imshow(I_3_dummy,[]);title('Ιmage Dark road 3 locally equalized');

subplot(2,3,4)
imhist(uint8(I_1_dummy));
subplot(2,3,5)
imhist(uint8(I_2_dummy));
subplot(2,3,6)
imhist(uint8(I_3_dummy));

print(gcf, '-dpng', 'Erotima4_png/erwtima4_3_Histograms_locallyequalized.png');
%subplot(2,3,2)
%imhist(B_2);
%subplot(2,3,3)
%imhist(B_3);

%showing images after local histogram equalization




% subplot(1,2,2)
% imshow(B_3,[]);title('original image dark_road_1 locally equalized with replicate padding');
%%
figure;
imshow(B_1);title('original image dark_road_2 locally equalized');


figure;
imshow(B_3);title('original image dark_road_3  locally equalized');


