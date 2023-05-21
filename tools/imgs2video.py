import imageio
import os
import numpy as np
import glob


img_paths = glob.glob('logs/merge_arm_over_female_sculpture/imgs_test_iters/*.png')

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# img_fn = [fn for fn in sorted(os.listdir(img_dir)) if fn.endswith('.png')]

imgs = []

for fn in img_paths:
    img = imageio.imread(fn)
    imgs.append(img)

imgs_np = np.stack(imgs)
imageio.mimwrite('logs/merge_arm_over_female_sculpture/imgs_test_iters/video.mp4', imgs_np, fps=30, quality=8)
