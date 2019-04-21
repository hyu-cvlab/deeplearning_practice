import os
import os.path as osp
from urllib.request import urlopen
import zipfile
import shutil
import copy
import sys

SEQ_SRC='./datasets/'
ATTR_LIST_FILE = 'attr_list.txt'
ATTR_DESC_FILE = 'attr_desc.txt'
TB_50_FILE = 'tb_50.txt'
TB_100_FILE = 'tb_100.txt'
CVPR_13_FILE = 'cvpr13.txt' 
ATTR_FILE = 'attrs.txt'
INIT_OMIT_FILE = 'init_omit.txt'
GT_FILE = 'groundtruth_rect.txt'
DOWNLOAD_URL = "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq_new/{0}.zip"
def download_sequence(seqName):
    file_name = SEQ_SRC + seqName + '.zip'

    if seqName == 'Jogging-1' or seqName == 'Jogging-2':
        url = DOWNLOAD_URL.format('Jogging')
        download_and_extract_file(url, file_name, SEQ_SRC)
        src = SEQ_SRC + 'Jogging/'
        dst1 = SEQ_SRC + 'Jogging-1/'
        dst2 = SEQ_SRC + 'Jogging-2/'
        if not os.path.exists(dst1 + 'img'):
            shutil.copytree(src + 'img', dst1 + 'img')
        if not os.path.exists(dst2 + 'img'):
            shutil.copytree(src + 'img', dst2 + 'img')
        shutil.move(src + 'groundtruth_rect.1.txt', dst1 + GT_FILE)
        shutil.move(src + 'groundtruth_rect.2.txt', dst2 + GT_FILE)
        shutil.move(src + 'jogging-1.txt', dst1 + INIT_OMIT_FILE)
        shutil.move(src + 'jogging-2.txt', dst2 + INIT_OMIT_FILE)
        shutil.rmtree(src)

    elif seqName == 'Skating2-1' or seqName == 'Skating2-2':
        url = DOWNLOAD_URL.format('Skating2')
        download_and_extract_file(url, file_name, SEQ_SRC)
        src = SEQ_SRC + 'Skating2/'
        dst1 = SEQ_SRC + 'Skating2-1/'
        dst2 = SEQ_SRC + 'Skating2-2/'
        if not os.path.exists(dst1 + 'img'):
            shutil.copytree(src + 'img', dst1 + 'img')
        if not os.path.exists(dst2 + 'img'):
            shutil.copytree(src + 'img', dst2 + 'img')
        shutil.move(src + 'groundtruth_rect.1.txt', dst1 + GT_FILE)
        shutil.move(src + 'groundtruth_rect.2.txt', dst2 + GT_FILE)
        shutil.rmtree(src)

    elif seqName == 'Human4-1' or seqName == 'Human4-2':
        url = DOWNLOAD_URL.format('Human4')
        download_and_extract_file(url, file_name, SEQ_SRC)
        src = SEQ_SRC + 'Human4/'
        # dst1 = SEQ_SRC + 'Human4-1/'
        dst2 = SEQ_SRC + 'Human4-2/'
        # if not os.path.exists(dst1 + 'img'):
        #     shutil.copytree(src + 'img', dst1 + 'img')
        if not os.path.exists(dst2 + 'img'):
            shutil.copytree(src + 'img', dst2 + 'img')
        # shutil.move(src + 'groundtruth_rect.1.txt', dst1 + GT_FILE)
        shutil.move(src + 'groundtruth_rect.2.txt', dst2 + GT_FILE)
        shutil.rmtree(src)

    else:
        url = DOWNLOAD_URL.format(seqName)
        download_and_extract_file(url, file_name, SEQ_SRC)
            
    if os.path.exists(SEQ_SRC + '__MACOSX'):
        shutil.rmtree(SEQ_SRC + '__MACOSX')
        

def download_and_extract_file(url, dst, ext_dst):  
    print('Connecting to {0} ...'.format(url))
    try:
        u = urlopen(url)
    except:
        print('Cannot download {0} : {1}'.format(
            url.split('/')[-1], sys.exc_info()[1]))
        sys.exit(1)
    f = open(dst, 'wb')
    meta = u.info()
    file_size = int(meta.get("Content-Length"))
    print("Downloading {0} ({1} Bytes)..".format(url.split('/')[-1], file_size))
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"{0:d} ({1:3.2f}%)".format(
            file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status,end="")
    f.close()

    f = open(dst, 'rb')
    z = zipfile.ZipFile(f)
    print('\nExtracting {0}...'.format(url.split('/')[-1]))
    z.extractall(ext_dst)
    f.close()
    os.remove(dst)


f = open('./datasets/tb_100.txt')
seq_list = f.readlines()
names = sorted([x.split('\t')[0].strip() for x in seq_list])
print(names)




for name in names:
    data_path = osp.join('./datasets', name)
    if not osp.exists(data_path):
        os.mkdir(data_path)
    if not osp.exists(osp.join(data_path, 'img')):
        download_sequence(name)