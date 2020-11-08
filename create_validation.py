from os.path import join, exists
from os import listdir, makedirs
from shutil import move
import random

classes = [
    "fruit",
    "flower",
    "both",
    "ambiguous",
]

train_dir = "/home/lindsey/Documents/Git_Projects/bird_species_classification/train/"
validation_dir = "/home/lindsey/Documents/Git_Projects/bird_species_classification/validation/"

#bird_specie -> classification
#bird_specie_counter -> classification_counter
#species -> classes

def create_validation():
    """Validation data sepration from augmented training images.
    Number of images chosen for validation depends upon the
    number of images present in the directory. If less than 78,
    then 6 images are moved into validation folder. Similarly,
    two if conditions for cases with less than 81 and greater
    than 85. Images are selected using random sampling.
    """
    for classification in classes:

        train_imgs_path = join(train_dir, classification)
        #if not exists(join(validation_dir, classification)):
        destination = join(validation_dir, classification)
            #destination = makedirs(join(validation_dir, classification))

        train_imgs = listdir(train_imgs_path)
        number = len(train_imgs)  # number of images in each category
        if number < 78:
            validation_separation = random.sample(train_imgs, 6)
            for img_file in validation_separation:
                move(join(train_imgs_path, img_file),
                     join(destination, img_file))

        elif 78 <= number <= 81:
            validation_separation = random.sample(train_imgs, 8)
            for img_file in validation_separation:
                move(join(train_imgs_path, img_file),
                     join(destination, img_file))

        elif number > 85:
            validation_separation = random.sample(train_imgs, 9)
            for img_file in validation_separation:
                move(join(train_imgs_path, img_file),
                     join(destination, img_file))


if __name__ == "__main__":
    create_validation()
