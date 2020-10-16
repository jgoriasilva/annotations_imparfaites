import os
import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
from skimage import io

img = io.imread(os.path.join('images','oubli','oubli_95_run_0.png'))
hist, bins_center = exposure.histogram(img)
threshold = filters.threshold_otsu(img)

plt.plot(bins_center, hist, lw=2)
plst.axvline(val, color='k', ls='--')
plt.tight_layout()
plt.show()


plt.imshow(img < threshold, cmap='gray')
plt.show()
