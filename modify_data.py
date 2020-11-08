from os.path import join, exists
from os import listdir, makedirs
from shutil import copyfile

classes = [
    "fruit",
    "flower",
    "both",
    "ambiguous",
]


source_folder = "/home/lindsey/Desktop/Propulsion/Data/Final_Project/brassica_photos/"
destination_folder = "/home/lindsey/Documents/Git_Projects/bird_species_classification/train/"

#bird_specie -> classification
#bird_specie_counter -> classification_counter
#species -> classes

def rename_files():
    """
    Initially the file names are incosistent. This function
    changes the file name to make it more understanding.

    Example - for example, DSC_6272.jpg may be changed to 100101.jpg
    For bird_specie_counter < 10, in this,
    100 -> original image, 1 -> Class Number, 01 -> Image Number

    Similarly, for the case if the species counter is greater than 10.
    """
    classification_counter = 1

    for classification in classes:

        #
        source_image_dir = join(source_folder, classification)
        print(source_image_dir)
        source_images = listdir(source_image_dir)
        print(source_images)

        for source_image in source_images:

            destination = join(destination_folder, classification)
            print(destination)
            if classification_counter < 10:

                images = 0
                for source_image in source_images:

                    if images < 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(100)
                                + str(classification_counter)
                                + str(0)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    elif images >= 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(100)
                                + str(classification_counter)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    images += 1

            elif classification_counter >= 10:

                images = 0

                for source_image in source_images:

                    if images < 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(10)
                                + str(classification_counter)
                                + str(0)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    elif images >= 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(10)
                                + str(classification_counter)
                                + str(images)
                                + ".jpg",
                            ),
                        )
                    images += 1

        classification_counter += 1


if __name__ == "__main__":
    for classification in classes:
        if not exists(join(destination_folder, classification)):
            destination = makedirs(join(destination_folder, classification))
    rename_files()
