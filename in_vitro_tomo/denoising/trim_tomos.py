#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
print('Imports finished. Beginning script...')
################################################################################
def trim_tomogram(file_name, crop_coords):
    x1,x2,y1,y2,z1,z2 = crop_coords
    tomo = []
    print('Opening tomo: %s'%file_name)
    output_ID = file_name.split('/')[-1][:5]
    print(output_ID)
    with mrcfile.open(file_name, 'r') as mrc:
        tomo = mrc.data
    
    print('Cropping micrograph section...')
    crop_box = tomo[z1:z2,y1:y2,x1:x2]
    crop_box = np.asarray(crop_box)
    print('Finished cropping micrograph.')
    print('Saving file %s:...'%str(output_ID))
    #bin2
    #with mrcfile.new(os.path.dirname(os.path.dirname(file_name))+'/cropped_tomos/'+output_ID+'_rec.mrc', overwrite=True) as mrc:
    # bin3
    #with mrcfile.new(os.path.dirname(os.path.dirname(file_name))+'/curated_tomos_bin3/'+output_ID+'_rec.mrc', overwrite=True) as mrc:
    #    mrc.set_data(np.asarray(crop_box).astype('float32'))
    # bin2 denoised
    with mrcfile.new('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/'+'/cryoCare_tomos_bin2/'+output_ID+'_rec.mrc', overwrite=True) as mrc:
        mrc.set_data(np.asarray(crop_box).astype('float32'))


################################################################################
# bin2
#file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/orig_tomos/*.mrc'))
# bin2 denoised
file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/khamilton/cryoCare_bin2-holes/denoise/*bin2_rec.mrc'))
#bin3
#file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/orig_tomos_bin3/*.mrc'))


#bin2 10p1 A/px
trim_coords = [[950, 1500, 290, 530], #488
               [946, 1448, 322, 443],#503
               [991, 1421, 280, 510],#506
               [946, 1328, 230, 530], #508
               [1183, 1259, 280, 515], #511
               [958, 1364, 300, 590], #524
               [1015, 1331, 180, 600], #525
               [1075, 1370, 270, 620], #532
               [950, 1360, 315, 550], # 535
               [1201, 1367, 305, 470], #541
               [1125, 1530, 250, 530], # 544
               [960, 1300, 320, 460], # 548
               [1105, 1478, 250, 525], # 583
               [860, 1440, 265, 450], # 584
               [1185, 1245, 300, 530], # 590
               [1000, 1400, 280, 500], # 707
               [970, 1360, 270, 450], # 709
               [1170, 1280, 320, 450], # 717
               [1150, 1400, 150, 550], #720
               [960, 1450, 150, 400], #721
               [1000, 1500, 260, 540], # 725
               [970, 1420, 290, 470], #740
               [1040, 1500, 300, 520], # 744
               [1000, 1450, 330, 480], #745
               [1000, 1440, 300, 470], #746
               [1040, 1480, 290, 400], #759
               [790, 1320, 290, 540],#765
               [900, 1200, 260, 460], # 779
               [1130, 1370, 320, 510], #780
               [1000, 1610, 270, 480], # 783
               [1000, 1320, 300, 490], #784
               [1050, 1390, 310, 460], # 812
               [1050, 1380, 220, 500], # 813
               [1050, 1470, 290, 530], # 816 bad
               [750, 1400, 290, 560], #818
               [850, 1450, 215, 550] # 819
               ]

#bin3 15p15 A/px
trim_coords2 = [[633,	1000,	193,	353], #488
               [631,	965,	215,	295],#503
               [661,	947,	187,	340],#506
               [631,	885,	153,	353], #508
               [639,	909,	200,	393], #524
               [677,	887,	120,	400], #525
               [633,	907,	210,	367], # 535
               [801,	911,	203,	313], #541
               [750,	1020,	167,	353], # 544
               [640,	867,	213,	307], # 548
               [667,	933,	187,	333], # 707
               [767,	933,	100,	367], #720
               [667,	1000,	173,	360], # 725
               [647,	947,	193,	313], #740
               [693,	1000,	200,	347], # 744
               [667,	967,	220,	320], #745
               [667,	960,	200,	313], #746
               [527,	880,	193,	360],#765
               [600,	800,	173,	307], # 779
               [753,	913,	213,	340], #780
               [567,	967,	143,	367] # 819
               ]

print(len(trim_coords))
#700 for bin2, 450 for bin3
for i in range(0, len(file_names)):
    trim_tomogram(file_names[i], [trim_coords[i][0]-700, trim_coords[i][0]+700, trim_coords[i][1]-700, trim_coords[i][1]+700,trim_coords[i][2], trim_coords[i][3]])
    #trim_tomogram(file_names[i], [trim_coords[i][0]-450, trim_coords[i][0]+450, trim_coords[i][1]-450, trim_coords[i][1]+450,trim_coords[i][2], trim_coords[i][3]])



