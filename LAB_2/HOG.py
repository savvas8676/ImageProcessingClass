import numpy as np
from utils import LOAD_YAML
from math import floor

class HOG:
    
    def __init__(self, input_image = None, cell_size=8, block_size=2, orientations=9, factor=1e-7):
        self.config  = LOAD_YAML()
        self.image = input_image
        self.cell_size = self.config['HOG']['patch'] #for each individual histogram
        self.block_size = self.config['HOG']['block_size'] #how many cells to be added incorporated
        self.orientations = self.config['HOG']['orientations']
        self.angle_unit = 180 // self.orientations #floor division
        self.epsilon = factor #simply to prevent division by zero in normalization

    def process(self):
        '''
            Function to implement the whole procedure of HOG feature exctraction for one image
        '''
        magnitude_image, angle_image = self.Gradient()
        histograms_matrix,blocksx,blocksy = self.calculate_histogram_matrix(magnitude_image, angle_image)
        feature  = self.block_normalization(histograms_matrix,blocksx,blocksy) #this is the feature vector for the image
        return feature
    
    def Gradient(self)->tuple:
        '''
            Function to implement the computation of the images gradient and angle images 
        '''
        #Step 1: Compute gradient images using numpy gradient for a faster implementation that does not use convolution
        gy, gx = np.gradient(self.image)

        #Step 2: Compute magnitude of gradient and angle image using np hypot
        magnitude_image = np.hypot(gx, gy)   # = np.sqrt(gx**2 + gy**2)
        angle_image = np.rad2deg(np.arctan2(gy, gx)) % 180 # Mod 180 because we use unsigned angle and -20 degrees is 160

        #Step 3: Returning gradient images
        return magnitude_image,angle_image
        
    def calculate_histogram_for_one_cell(self,magnitude_image_patch:np.array,angle_image_patch:np.array)->None:
        '''
            Function to implement the calculation of the histogram  for one cell an orientation-lengthed vector
            Input: Magnitude and angle image of a single patch
            Output: hog , numpy array vector of histogram bins
        '''
        bin_width = int(180 / self.orientations)
        hog = np.zeros(self.orientations)
        for i in range(angle_image_patch.shape[0]):
            for j in range(angle_image_patch.shape[1]):
                orientation = angle_image_patch[i, j]
                lower_bin_idx = int(orientation / bin_width)
                if self.config['report']['bilinear']:
                    lower_bin_idx = floor(orientation / bin_width- 1/2) % self.orientations
                    upper_bin_idx = (lower_bin_idx+1)% self.orientations
                    centre_upper = 0 + (upper_bin_idx)*bin_width + bin_width/2
                    centre_lower = 0 + (lower_bin_idx)*bin_width + bin_width/2

                    #update magnitude
                    hog[lower_bin_idx] += magnitude_image_patch[i, j] * ((centre_upper - orientation)/bin_width)
                    hog[upper_bin_idx] += magnitude_image_patch[i, j] * ((orientation - centre_lower)/bin_width)
                else:

                    hog[lower_bin_idx] += magnitude_image_patch[i, j] #--------------------------------------------------------------!

        return hog# / (magnitude_image_patch.shape[0] * magnitude_image_patch.shape[1]) # Normalizing histogram as per sci-kit image with the number of elements in the patch


    def normalize_vector(self,v)->np.array:
        """
        Input: v numpy array containing vector to be normalized according to L2 normalization scheme as per the paper
        Return a normalized vector
        """
        # epsilon is used to prevent zero divide exceptions (in case v is zero)
        return v / np.sqrt(np.sum(v ** 2) + self.epsilon ** 2) 

    def calculate_histogram_matrix(self,magnitude_image:np.array, angle_image:np.array)->np.array:
        '''
            Function to implement the calculation of the total histogram matrix for all patches in the image
            Input: Magnitude and angle image 
            Output: hog , numpy array matrix of histogra vectors 3D
        '''

        sx, sy = magnitude_image.shape
        cx, cy = self.cell_size,self.cell_size
        bx, by = self.block_size,self.block_size

    
        n_cellsx = int(sx / cx) # Number of cells in x axis rounding to the smallest integer
        n_cellsy = int(sy / cy) # Number of cells in y axis rounding to the smallest integer

        n_blocksx = int(n_cellsx - bx) + 1 #Number of blocks calculated based on cells 
        n_blocksy = int(n_cellsy - by) + 1 #Number of blocks calculated based on cells 

        hog_cells = np.zeros((n_cellsx, n_cellsy, self.orientations))

        prev_x = 0
        # Compute HOG of each cell
        for it_x in range(n_cellsx):
            prev_y = 0
            for it_y in range(n_cellsy):
                magnitudes_patch = magnitude_image[prev_x:prev_x + cx, prev_y:prev_y + cy] #upper limit is excluded
                orientations_patch = angle_image[prev_x:prev_x + cx, prev_y:prev_y + cy]

                hog_cells[it_x, it_y,:] = self.calculate_histogram_for_one_cell(magnitudes_patch, orientations_patch)

                prev_y += cy
            prev_x += cx

        return hog_cells,n_blocksx,n_blocksy

        

    def block_normalization(self,histogram_matrix:np.array,n_blocksx,n_blocksy)->np.array:
        '''
            Function to perform block normalization and compute the feature vector
            Input: histogram_matrix numpy array that contains in each cell a 9-bin histogram
            Output: feature_vector finally computed

        '''
       
        hog_blocks_normalized = np.zeros((n_blocksx, n_blocksy, self.block_size * self.block_size *self.orientations))

        # Normalize HOG by block
        for it_blocksx in range(n_blocksx):
            for it_blocksy in range(n_blocksy):
                hog_block = histogram_matrix[it_blocksx:it_blocksx + self.block_size, it_blocksy:it_blocksy + self.block_size].ravel()
                hog_blocks_normalized[it_blocksx, it_blocksy,:] = self.normalize_vector(hog_block)

        feature_vector = hog_blocks_normalized.ravel()
        if self.config['report']['bilinear']:
            norm_feature_vector = self.normalize_vector(feature_vector)
            clipped_feature_vector = np.clip(norm_feature_vector, None,0.2)
            feature_vector = self.normalize_vector(clipped_feature_vector)
        else:
            pass

        return feature_vector
