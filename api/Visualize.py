import matplotlib.pyplot as plt
import numpy

def show_sample_dataset(loaded_data, mean, std):
    images = next(iter(loaded_data))
    
    imgs = images[0][:5].numpy()
    labels = images[1][:5]
    
    count, c = 0, 0
    
    fig, ax = plt.subplots(1,5,figsize = (14,14))
    fig.tight_layout()
    
    print("Image shape: ",imgs[0].shape, imgs[1].shape)
    print(type(labels[0]))
    
    for (im, lbl) in zip(imgs, labels):

        #denormalizing images
        for i in range(im.shape[0]):
            im[i] = (im[i] * std[i]) + mean[i]

        im = numpy.transpose(im, (1,2,0))
        
        ax[count].imshow(im)
        ax[count].axis("off")
        ax[count].set_title(lbl)

        count += 1