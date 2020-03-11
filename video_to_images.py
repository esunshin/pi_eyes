image_suffix = ".jpg" # Don't change

def format_directory(directory_name=None):
    return "./" + ("" if directory_name is None else directory_name + "/")

def get_filename(n, prefix=None):
    return (prefix + "_" if prefix is not None else "") + str(n) + image_suffix

def extract_images(video_filename, write_directory=None, prefix=None):
    import cv2
    vidcap = cv2.VideoCapture(video_filename)
    success,image = vidcap.read()
    count = 0
    while success:
        img_name = format_directory(write_directory) + get_filename(count, prefix)
        cv2.imwrite(img_name, image)    # save frame as `image_suffix` file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return count

def get_images(read_directory=None, prefix=None):
    import os
    folder = format_directory(read_directory)
    return os.listdir(folder).sort() if prefix is None \
                    else [item for item in os.listdir(folder) if item[:len(prefix)] == prefix].sort() # putting os.listdir(folder) inside might not work
 
# saves every n-th image, deletes the rest
def drop_images(n, read_directory=None, prefix=None):
    import os
    directory = format_directory(read_directory)

    all_files = get_images(read_directory, prefix)

    for item, count in zip(all_files, range(n)):
        if count % n != 0:
            filename = directory + item
            assert os.path.isfile(filename), filename + " is not a file"
            os.unlink(filename)

def rename_images(read_directory=None, prefix=None):
    import os
    directory = format_directory(read_directory)
    
    count = 0
    for image in get_images(read_directory, prefix): # may need to move out get_images(...)
        source = directory + image
        assert os.path.isfile(source), source + " is not a file"
        assert "." + image.split['.'][1] == image_suffix, "File is not image: " + image
        
        new_filename = directory + get_filename(count, prefix)
        os.rename(source, new_filename)
    
        count += 1
    
    return count


