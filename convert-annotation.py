import os
import glob
from PIL import Image
from pathlib import Path

path = Path('data/annotations')

source = 'training-ori'
path_src = path/source
path_src.mkdir(exist_ok=True)

gt= 'training'
path_gt = path/gt
path_gt.mkdir(exist_ok=True)

filenames = [img for img in glob.glob(str(path_src/"*.*"))]

filenames.sort()

for img in filenames:
    img_path = f'{img}'
    img_split = Path(img_path).stem
    print(img_split)

    im = Image.open(img_path)
    pixelMap = im.load()

    imgL = Image.new( 'L', im.size)
    pixelsNewL = imgL.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            
            if pixelMap[i,j] == 255 :
                pixelsNewL[i,j] = 2
            else:
                pixelsNewL[i,j] = 1
                
    im.close()    
    imgL.save(os.path.join(path_gt, img_split+".png")) 
    imgL.close()
