# This file will be the official reference to define all of the functions we will use throughout the dataset
# I guess I need to do the imports from here?
import os
import numpy as np
import dicom
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Newer imports
from skimage.morphology import binary_closing, disk, binary_erosion, ball
from skimage.segmentation import clear_border
from skimage.filters import roberts


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

# Convert image to HU units
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image
    
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
    
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def ext_mask(binary_image, itrs=1):
    return scipy.ndimage.morphology.binary_dilation(binary_image, iterations=itrs).astype(binary_image.dtype)
    
def Nget_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = scipy.ndimage.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return im
def Nsegment_lung_from_ct_scan(ct_scan):
    return np.asarray([Nget_segmented_lungs(slice) for slice in ct_scan])
    
def Nplot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()  

## UNET Candidate Detection
#
#'''
#This funciton reads a '.mhd' file using SimpleITK and return the image array, 
#origin and spacing of the image.
#'''
#def load_itk(filename):
#    # Reads the image using SimpleITK
#    itkimage = sitk.ReadImage(filename)
#    
#    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
#    ct_scan = sitk.GetArrayFromImage(itkimage)
#    
#    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
#    origin = np.array(list(reversed(itkimage.GetOrigin())))
#    
#    # Read the spacing along each dimension
#    spacing = np.array(list(reversed(itkimage.GetSpacing())))
#    
#    return ct_scan, origin, spacing
#
#'''
#This function is used to convert the world coordinates to voxel coordinates using 
#the origin and spacing of the ct_scan
#'''
#def world_2_voxel(world_coordinates, origin, spacing):
#    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
#    voxel_coordinates = stretched_voxel_coordinates / spacing
#    return voxel_coordinates
#
#'''
#This function is used to convert the voxel coordinates to world coordinates using 
#the origin and spacing of the ct_scan.
#'''
#def voxel_2_world(voxel_coordinates, origin, spacing):
#    stretched_voxel_coordinates = voxel_coordinates * spacing
#    world_coordinates = stretched_voxel_coordinates + origin
#    return world_coordinates
#
#def seq(start, stop, step=1):
#	n = int(round((stop - start)/float(step)))
#	if n > 1:
#		return([start + step*i for i in range(n+1)])
#	else:
#		return([])
#
#'''
#This function is used to create spherical regions in binary masks
#at the given locations and radius.
#'''
#def draw_circles(image,cands,origin,spacing):
#      #make empty matrix, which will be filled with the mask
#      RESIZE_SPACING = [1, 1, 1]
#      image_mask = np.zeros(image.shape)
#
#	#run over all the nodules in the lungs
#	for ca in cands.values:
#		#get middel x-,y-, and z-worldcoordinate of the nodule
#		radius = np.ceil(ca[4])/2
#		coord_x = ca[1]
#		coord_y = ca[2]
#		coord_z = ca[3]
#		image_coord = np.array((coord_z,coord_y,coord_x))
#
#		#determine voxel coordinate given the worldcoordinate
#		image_coord = world_2_voxel(image_coord,origin,spacing)
#
#		#determine the range of the nodule
#		noduleRange = seq(-radius, radius, RESIZE_SPACING[0])
#
#		#create the mask
#		for x in noduleRange:
#			for y in noduleRange:
#				for z in noduleRange:
#					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
#					if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
#						image_mask[np.round(coords[0]),np.round(coords[1]),np.round(coords[2])] = int(1)
#	
#	return image_mask
#
#'''
#This function takes the path to a '.mhd' file as input and 
#is used to create the nodule masks and segmented lungs after 
#rescaling to 1mm size in all directions. It saved them in the .npz
#format. It also takes the list of nodule locations in that CT Scan as 
#input.
#'''
#def create_nodule_mask(imagePath, maskPath, cands):
#	#if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
#	img, origin, spacing = load_itk(imagePath)
#
#	#calculate resize factor
#    RESIZE_SPACING = [1, 1, 1]
#	resize_factor = spacing / RESIZE_SPACING
#	new_real_shape = img.shape * resize_factor
#	new_shape = np.round(new_real_shape)
#	real_resize = new_shape / img.shape
#	new_spacing = spacing / real_resize
#	
#	#resize image
#	lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
#    
#    # Segment the lung structure
#	lung_img = lung_img + 1024
#	lung_mask = segment_lung_from_ct_scan(lung_img)
#	lung_img = lung_img - 1024
#
#	#create nodule mask
#	nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)
#
#	lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros((lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))
#
#	original_shape = lung_img.shape	
#	for z in range(lung_img.shape[0]):
#		offset = (512 - original_shape[1])
#		upper_offset = np.round(offset/2)
#		lower_offset = offset - upper_offset
#
#		new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)
#
#		lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
#		lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
#		nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]
#
#    # save images.    
#	np.save(imageName + '_lung_img.npz', lung_img_512)
#	np.save(imageName + '_lung_mask.npz', lung_mask_512)
#	np.save(imageName + '_nodule_mask.npz', nodule_mask_512)   
#
## change the loss function
#def dice_coef(y_true, y_pred):
#    smooth = 1.
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
#def dice_coef_loss(y_true, y_pred):
#	return -dice_coef(y_true, y_pred)
#
#'''
#The UNET model is compiled in this function.
#'''
#def unet_model():
#	inputs = Input((1, 512, 512))
#	conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
#	conv1 = Dropout(0.2)(conv1)
#	conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
#	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#	conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
#	conv2 = Dropout(0.2)(conv2)
#	conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
#	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#	conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
#	conv3 = Dropout(0.2)(conv3)
#	conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
#	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#	conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
#	conv4 = Dropout(0.2)(conv4)
#	conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
#	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#	conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
#	conv5 = Dropout(0.2)(conv5)
#	conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)
#
#	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#	conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
#	conv6 = Dropout(0.2)(conv6)
#	conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)
#
#	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#	conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
#	conv7 = Dropout(0.2)(conv7)
#	conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)
#
#	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#	conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
#	conv8 = Dropout(0.2)(conv8)
#	conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)
#
#	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
#	conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
#	conv9 = Dropout(0.2)(conv9)
#	conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)
#
#	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
#
#	model = Model(input=inputs, output=conv10)
#	model.summary()
#	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#
#	return model