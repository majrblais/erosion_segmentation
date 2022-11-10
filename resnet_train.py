import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras_segmentation.models.unet import resnet50_unet
model = resnet50_unet(n_classes=2 ,  input_height=1024, input_width=1024)
model.train(train_images="./new_t/images/data/",train_annotations="./new_t/masks/data/",validate=True,val_images="./valid_data/images/data/",val_annotations="./valid_data/masks/data/",checkpoints_path="./b/",epochs=100)
