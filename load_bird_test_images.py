import pickle
import os
from PIL import Image

def load_filenames(data_dir):
    filepath = '%s/filenames.pickle' % (data_dir)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    else:
        filenames = []
    return filenames


if __name__ == '__main__':
    src_imgs_path = r'data\birds\CUB_200_2011\images'
    dest_imgs_path = r'data\birds\test\images'
    test_folder_path = r'data\birds\test'

    if not os.path.isdir(dest_imgs_path):
        os.mkdir(dest_imgs_path)

    filenames = load_filenames(test_folder_path)
    for idx, full_fname in enumerate(filenames):
        splitted_full_fname = full_fname.split('/')
        class_name = splitted_full_fname[0]
        fname = splitted_full_fname[1]

        src_file_path = os.path.join(os.path.join(src_imgs_path, class_name), fname+'.jpg')
        dest_file_path = os.path.join(dest_imgs_path, f'img_{idx}.png')
        img = Image.open(src_file_path)
        img = img.resize((256, 256))
        img.save(dest_file_path)
