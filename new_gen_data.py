# First pass data general statistics

import definitions
import time
import pandas as pd
import json
import mrcfile

# Load Patient Folder
INPUT_FOLDER = 'C:/Users/stephen_GAME/Google Drive/UB documents/Spring 2017/BE 400/sample_images/'
#INPUT_FOLDER = 'E:/BE400_Data/stage1/'  # Note, this is for sample images
patients = definitions.os.listdir(INPUT_FOLDER)
patients.sort()
# Load Patient Labels
patientLabels = pd.read_csv('C:/Users/stephen_GAME/Google Drive/UB documents/Spring 2017/BE 400/kaggle_group/stage1_labels.csv')
dicPatientLabels = patientLabels.set_index('id').to_dict()

# Make list of patient lung characteristics
patientID           = []
patientDiag         = []
voxSizes            = []
voxDims1            = []
voxDims2            = []
voxDims3            = []
LungVolumes         = []
UsedVolumes         = []
meanInsideVolumes   = []
stdInsideVolumes    = []
procTimes           = []

histCount1          = []
histCount2          = []
histCount3          = []
histCount4          = []
histCount5          = []
histCount6          = []
histCount7          = []
histCount8          = []
histCount9          = []
histCount10         = []
histCount11         = []
histCount12         = []
histCount13         = []
histCount14         = []
histCount15         = []

Ttime = 0
for idx, patient in enumerate(patients):
#for idx, patient in enumerate(patientLabels.loc[:,'id']):
    start       = time.perf_counter()
    data        = definitions.load_scan(INPUT_FOLDER + patient)
    pixels      = definitions.get_pixels_hu(data)
    npixels     = definitions.resample(pixels,data, [1,1,1])[0]
    voxDim      = definitions.np.ndarray.tolist(definitions.np.array([data[0].SliceThickness] + data[0].PixelSpacing, dtype=definitions.np.float32))
    voxSize     = definitions.np.prod(voxDim)
    lungVolume  = definitions.segment_lung_mask(npixels, True)
    extVolume   = definitions.ext_mask(lungVolume,2)
    smVolume    = definitions.scipy.ndimage.morphology.binary_erosion(lungVolume,iterations=10)
    outter_vals = extVolume == 0
    innerVals   = smVolume == 1
    npixels[outter_vals] = 0
    npixels[innerVals] = 0    
    meanInsideVolume  = definitions.np.mean( npixels[npixels != 0] )
    stdInsideVolume   = definitions.np.std(  npixels[npixels != 0] )
    # New Image Properties
    histCounts  = definitions.np.histogram(npixels, bins=15, density = True )[0]
   # entr        = definitions.scipy.stats.entropy([npixels != 0].flatten())

   
    patientID.append(patient)
#    patientDiag.append( patientLabels.at[patient, 'cancer'] )
    try:
        patientDiag.append( dicPatientLabels['cancer'][patient] )
    except:
        patientDiag.append( 2 )
    voxSizes.append(voxSize)
    voxDims1.append(voxDim[0])    
    voxDims2.append(voxDim[1])    
    voxDims3.append(voxDim[2])
    volumes.append(definitions.np.sum(lungVolume))
    UsedVolumes.append(definitions.np.sum(extVolume) - definitions.np.sum(smVolume))
    meanInsideVolumes.append(meanInsideVolume)
    stdInsideVolumes.append(stdInsideVolume)
    
    histCount1.append(histCounts[0])
    histCount2.append(histCounts[1])
    histCount3.append(histCounts[2])
    histCount4.append(histCounts[3])
    histCount5.append(histCounts[4])
    histCount6.append(histCounts[5])
    histCount7.append(histCounts[6])
    histCount8.append(histCounts[7])
    histCount9.append(histCounts[8])
    histCount10.append(histCounts[9])
    histCount11.append(histCounts[10])
    histCount12.append(histCounts[11])
    histCount13.append(histCounts[12])
    histCount14.append(histCounts[13])
    histCount15.append(histCounts[14])
        
    procTimes.append(time.perf_counter() - start)
    Ttime = Ttime + procTimes[-1]
    print('Iteration: ', idx, 'Elapsed time: ', Ttime)
    fname = 'vols/img_' + str(idx) + '.mrc'
    with mrcfile.new(fname) as mrc:
        mrc.set_data(npixels)


print('It ran!')
# Write out using pandas
#I Guess use a dictionaary
genData = {}
genData['Patient ID'] = patientID
genData['voxSizes'] = voxSizes
genData['voxDims1'] = voxDims1
genData['voxDims2'] = voxDims2
genData['voxDims3'] = voxDims3
genData['volumes'] = volumes
genData['meanInsideVolumes'] = meanInsideVolumes
genData['stdInsideVolumes'] = stdInsideVolumes
genData['procTimes'] = procTimes
genData['histCount1'] = histCount1
genData['histCount2'] = histCount2
genData['histCount3'] = histCount3
genData['histCount4'] = histCount4
genData['histCount5'] = histCount5
genData['histCount6'] = histCount6
genData['histCount7'] = histCount7
genData['histCount8'] = histCount8
genData['histCount9'] = histCount9
genData['histCount10'] = histCount10
genData['histCount11'] = histCount11
genData['histCount12'] = histCount12
genData['histCount13'] = histCount13
genData['histCount14'] = histCount14
genData['histCount15'] = histCount15

df = pd.DataFrame( genData )
writer = pd.ExcelWriter('new_gen_data_stat.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='General Stats')
writer.save()

# To check, lets save all of the data as a JSON file
#with open('gen_data_stat.json', 'w') as f2:
#    json.dump(genData, f2)
print('well, there you have it, it ran!')