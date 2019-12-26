"""
objective :
author(s) : Ashwin de Silva
date      : 
"""

import nibabel as nib
import numpy as np

input_image = '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/itk/build/mask.nii.gz'

# read imagen
image = nib.load(input_image)
I = np.asarray(image.get_data(caching='unchanged'))
print('Image Shape : ', I.shape)

out = np.zeros((55, 55, 55, 65))

for i in range(65):
    out[..., i] = I


out = nib.Nifti1Image(out.astype(np.float32), image.affine)
nib.save(out, '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm/itk/build/mask_65.nii.gz')

