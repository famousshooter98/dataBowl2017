# This will use the UNET approach

import definitions

# Load Patient Folder
INPUT_FOLDER = 'C:/Users/stephen_GAME/Google Drive/UB documents/Spring 2017/BE 400/kaggle_group/sample_images/'  # Note, this is for sample images
patients = definitions.os.listdir(INPUT_FOLDER)
patients.sort()
for patient in patients:
    pat = definitions.load_scan(INPUT_FOLDER + patient)
    pixels = definitions.get_pixels_hu(pat)
    Nsegment = definitions.Nsegment_lung_from_ct_scan(pixels)
    Nsegment[Nsegment < 604] = 0
    
    ############################################
    
    selem = definitions.ball(2)
    binary = definitions.binary_closing(Nsegment, selem)
    
    label_scan = definitions.measure.label(binary)
    
    areas = [r.area for r in definitions.measure.regionprops(label_scan)]
    areas.sort()
    
    for r in definitions.measure.regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000
        
        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)
            
            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                Nsegment[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
    definitions.Nplot_3d(Nsegment,604)