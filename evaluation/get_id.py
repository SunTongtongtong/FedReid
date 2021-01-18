#
# Get camera and identity labels
#

import os

def get_id(img_path):
    camera_id = []
    labels = []

    if img_path[0][0].split('/')[6] in ['cuhk03-np','msmt17']:
        for path, v in img_path:
            filename = os.path.basename(path)
            label = int(path.split('/')[-2])
            if label == -1:
                labels.append(-1)
            else:
                labels.append(label)
            if path.split('/')[-5] == 'msmt17':
                camera = filename.split('_')[2]
            elif path.split('/')[-5] == 'cuhk03-np':
                camera = filename.split('_')[-2]
            else:
                assert "Get id function not correct"
            camera_id.append(int(camera))
        return camera_id,labels
    else:
        for path, v in img_path:
            filename = os.path.basename(path)
            label = filename[0:4] # filename[0:5] in CUHK-SYSU person Search
            camera = filename.split('c')[1]

            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0])) # camera[0:2] in MSMT17 & CUHK-SYSU person Search
        return camera_id, labels

#for cuhk03-np and msmt17
# def get_id(img_path):
#     camera_id = []
#     labels = []
#
    for path, v in img_path:
        filename = os.path.basename(path)
        label = int(path.split('/')[-2])
        if label==-1:
            labels.append(-1)
        else:
            labels.append(label)
        if path.split('/')[-5] == 'msmt17':
            camera = filename.split('_')[2]
        elif path.split('/')[-5] == 'cuhk03-np':
            camera = filename.split('_')[-2]
        else:
            assert "Get id function not correct"
        camera_id.append(int(camera))
    return camera_id,labels



