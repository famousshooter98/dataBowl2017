# First pass data general statistics

import definitions
import time
import pandas as pd
import json

# Load Patient Folder
INPUT_FOLDER = 'C:/Users/stephen_GAME/Google Drive/UB documents/Spring 2017/BE 400/kaggle_group/sample_images/'  # Note, this is for sample images
patients = definitions.os.listdir(INPUT_FOLDER)
patients.sort()

# Make list of patient lung characteristics
patientID           = []
voxSizes            = []
voxDims1            = []
voxDims2            = []
voxDims3            = []
volumes             = []
extVolumes          = []
meanInsideVolumes   = []
stdInsideVolumes    = []
NmeanInsideVolumes   = []
NstdInsideVolumes    = []
procTimes           = []



for idx, patient in enumerate(patients[0:5]):
    start       = time.perf_counter()
    data        = definitions.load_scan(INPUT_FOLDER + patient)
    pixels      = definitions.get_pixels_hu(data)
    voxDim      = definitions.np.ndarray.tolist(definitions.np.array([data[0].SliceThickness] + data[0].PixelSpacing, dtype=definitions.np.float32))
    voxSize     = definitions.np.prod(voxDim)
    lungVolume  = definitions.segment_lung_mask(pixels, True)
    extVolume   = definitions.ext_mask(lungVolume)
    nrmPixels   = definitions.zero_center(definitions.normalize(pixels))
    meanInsideVolume  = definitions.np.mean( pixels * lungVolume )
    stdInsideVolume   = definitions.np.std(  pixels * lungVolume )
    NmeanInsideVolume = definitions.np.mean( nrmPixels * lungVolume )
    NstdInsideVolume  = definitions.np.std(  nrmPixels * lungVolume )  
        
    patientID.append(patient)
    voxSizes.append(voxSize)
    voxDims1.append(voxDim[0])    
    voxDims2.append(voxDim[1])    
    voxDims3.append(voxDim[2])
    volumes.append(definitions.np.sum(lungVolume)*voxSize)
    extVolumes.append(definitions.np.sum(extVolume)*voxSize)
    meanInsideVolumes.append(meanInsideVolume)
    stdInsideVolumes.append(stdInsideVolume)
    NmeanInsideVolumes.append(NmeanInsideVolume)
    NstdInsideVolumes.append(NstdInsideVolume)
    procTimes.append(time.perf_counter() - start)
    print('Iteration: ', idx, 'Elapsed time: ', definitions.np.sum(procTimes))


# Write out using pandas
#I Guess use a dictionaary
genData = {}
genData['Patient ID'] = patientID
genData['voxSizes'] = voxSizes
genData['voxDims1'] = voxDims1
genData['voxDims2'] = voxDims2
genData['voxDims3'] = voxDims3
genData['volumes'] = volumes
genData['extVolumes'] = extVolumes
genData['meanInsideVolumes'] = meanInsideVolumes
genData['stdInsideVolumes'] = stdInsideVolumes
genData['procTimes'] = procTimes
genData['NmeanInsideVolumes'] = NmeanInsideVolumes
genData['NstdInsideVolumes'] = NstdInsideVolumes

df = pd.DataFrame( genData )
writer = pd.ExcelWriter('gen_data_stat.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='General Stats')
writer.save()

# To check, lets save all of the data as a JSON file
with open('gen_data_stat.json', 'w') as f2:
    json.dump(genData, f2)
print('well, there you have it, it ran!')

#Also, Old parts of code, (STUPID)

#        writer.writerows([patient + str(idx) + str(voxSize) + str(voxDim) + str(volumes[-1]) + str(extVolumes[-1]) + str(meanInsideVolume) + str(stdInsideVolume) + str(procTimes[-1])])
#        writer.writerows([patient, idx, voxSize, voxDim, volumes[-1], extVolumes[-1], meanInsideVolume, stdInsideVolume, procTimes[-1]])
#        writer.writerows([patient + idx + voxSize + voxDim + volumes[-1] + extVolumes[-1] + meanInsideVolume + stdInsideVolume + procTimes[-1]])
#        genData[patient] = [idx + voxSize + voxDim + volumes[-1] + extVolumes[-1] + meanInsideVolume + stdInsideVolume + procTimes[-1]]
#with open('gen_data_stat.csv', 'w', newline='') as f:
#    writer = csv.writer(f, dialect=csv.excel_tab)