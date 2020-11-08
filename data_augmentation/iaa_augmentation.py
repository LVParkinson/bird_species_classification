import cv2
from os.path import join
import os
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(5)

augmented_image_dir = "/home/lindsey/Documents/Git_Projects/bird_species_classification/train/"

#species -> classes
#bird_specie_counter -> classification_counter
#bird_specie_number -> classification_number

classes = [
    #"fruit",
    "flower",
    "both",
    "ambiguous",
]


""" Naming conventions can be different. This is
what I've used at my time. I just followed the table
present to generate that much number of images.

Type of Augmentation:
10 - Normal Image
20 - Gaussian Noise - 0.1* 255
30 - Gaussian Blur - sigma - 3.0
40 - Flip - Horizaontal
50 - Contrast Normalization - (0.5, 1.5)
60 - Hue
70 - Crop and Pad

Flipped
11 - Add - 2,3,4,5,6,12,13,14      7, 15, 16
12 - Multiply - 2,3,4,5,6,12,13,14 7, 15, 16
13 - Sharpen
14 - Gaussian Noise - 0.2*255
15 - Gaussian Blur - sigma - 0.0-2.0
16 - Affine Translation 50px x, y
17 - Hue Value
"""


def save_images(
    augmentated_image,
    destination,
    number_of_images,
    classification_counter,
    types
):

    image_number = str(number_of_images)
    number_of_images = int(number_of_images)

    if classification_counter < 10:

        if number_of_images < 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(0)
                    + str(classification_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )

        elif number_of_images >= 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(0)
                    + str(classification_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )

    elif classification_counter >= 10:

        if number_of_images < 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(classification_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )

        elif number_of_images >= 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(classification_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )


# Dataset Augmentation

gauss = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.2 * 255),
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
    iaa.AddToHueAndSaturation((-60, 60)),
    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.7, 1.3)),
    iaa.Affine(rotate=(-25, 25)),
    iaa.Fliplr(1.0)
], random_order=True)
    
    
# blur = iaa.GaussianBlur(sigma=(3.0))
# flip = iaa.Fliplr(1.0)
# contrast = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
sharp = iaa.Sequential([
    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.7, 1.3)),
    iaa.Affine(rotate=(-25, 25)),
    iaa.Fliplr(1.0)
], random_order = True)

affine = iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)})
# add = iaa.Add((-20, 20), per_channel=0.5)
# multiply  = iaa.Multiply((0.8, 1.2), per_channel=0.5)

hue = iaa.Sequential(
    [
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
    ]
)

aug = iaa.Sequential(
    [
        iaa.Fliplr(1.0),
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
    ]
)

def main ():
    for classification in classes:
        augmented_image_folder = join(augmented_image_dir, classification)
        source_images = os.listdir(augmented_image_folder)
        print(source_images)
        source_images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        
        file_count = sum(len(files) for _, _, 
            files in os.walk(augmented_image_folder))
        
        augmented_images_arr = []
        img_number = []
        classification_number = source_images[0]
        classification_number = int(classification_number[2:4])
        
        for source_image in source_images:
            if int(source_image[0]) == 1:
                img_number.append(source_image[4:6])
                img_path = join(augmented_image_folder, source_image)
                img = cv2.imread(img_path)
                augmented_images_arr.append(img)
                
        counter = 0
        if len(os.listdir(augmented_image_folder)) < 80:
            # Applying Gaussian image augmentation
            for augmented_image in sharp.augment_images(augmented_images_arr):
                save_images(
                    augmented_image,
                    augmented_image_folder,
                    img_number[counter],
                    classification_number,
                    51,
                )
                counter += 1
        

if __name__ == "__main__":
    main()
