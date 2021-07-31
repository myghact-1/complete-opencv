# https://en.wikipedia.org/wiki/High-dynamic-range_imaging


import cv2
import numpy as np
from glob import glob


#* 1. Loading exposure images into a list

images = glob("./images/hdr*")
images = [cv2.imread(image) for image in images]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)


#* 2. Merge exposures into HDR image

merge_debevec = cv2.createMergeDebevec()
hdr_debevec = merge_debevec.process(images, times=exposure_times.copy())
merge_robertson = cv2.createMergeRobertson()
hdr_robertson = merge_robertson.process(images, times=exposure_times.copy())


#* 3. Tonemap HDR image

tonemap1 = cv2.createTonemap(gamma=1.5)
res_debevec = tonemap1.process(hdr_debevec.copy())
res_robertson = tonemap1.process(hdr_robertson.copy())

#* 4. Merge exposures using Mertens fusion

merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(images)


#* 5. Convert to 8-bit and save

res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv2.imwrite("ldr_debevec.jpg", res_debevec_8bit)
cv2.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv2.imwrite("fusion_mertens.jpg", res_mertens_8bit)
