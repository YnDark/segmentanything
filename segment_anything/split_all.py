import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def main(img_path,sam_checkpoint,model_type,savepath):

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    print(masks)

    i = 1
    for mask in masks:
        mask = mask['segmentation']
        mask = ~mask
        mask = mask + 255
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = mask.astype(np.uint8)
        res = cv2.bitwise_and(image, mask)
        res[res == 0] = 255
        plt.imshow(res)
        plt.axis('off')
        plt.ylabel = [""]
        plt.xlabel = [""]
        plt.savefig(savepath+'/res-{}.png'.format(i + 2))
        plt.show()
        i = i+1
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main("J118m1-.png")
