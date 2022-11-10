#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import matplotlib.pyplot as plt
import random
import numpy
import time
from os import listdir
from os.path import isfile, join


# In[11]:

im='./train/images/'
mk='./train/masks/'
onlyimg = [f for f in listdir(im) if isfile(join(im, f))]
onlymask = [f for f in listdir(mk) if isfile(join(mk, f))]

from random import randrange

print('test')
#In[13]:
import time
k=0
#onlyimg=onlyimg[16:]
#onlymask=onlymask[16:]
#print
#onlymask=onlymask[:15]
#timeout = time.time() + 60
for i in range(len(onlyimg)):
    timeout = time.time() + 150
    img = cv2.imread(im+onlyimg[i],-1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(onlyimg[i])
    print(onlymask[i])
    #print("Y")
    #y_crop=int(input())
    #y_crop_2=int(input())
    #print("X")
    #x_crop=int(input())
    #x_crop_2=int(input())
    #print("K")
    h=150
    #h=int(input())i
    #img_rgb=im_rgb[y_crop:y_crop_2,x_crop:x_crop_2,:]


    mask = cv2.imread(mk+onlymask[i])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #mask=mask[y_crop:y_crop_2,x_crop:x_crop_2]

    print(onlyimg[i])
    print(onlymask[i])

    print("New image:" + str(i))
    j=0
    while True:
        #sleep(1)
        x=random.randint(1024,img_rgb.shape[0])
        y=random.randint(1024,img_rgb.shape[1])

        #mask_2=mask[x-512:x,y-512:y]
        try:
            mask_2=mask[x-1024:x,y-1024:y]
            img_2=img[x-1024:x,y-1024:y]
            

            unique, counts = numpy.unique(mask_2, return_counts=True)
            values=dict(zip(unique, counts))
            
            #unique2, counts2=numpy.unique(img_2,return_counts=True)
            #values2=dict(zip(unique2,counts2))

                
            #print(values2)

            if values[0]>=2000 and values[1]>=2000:
                j+=1
                k+=1
                print("Created Image: "+str(j))

                if j%5==0:
                    img_2=cv2.rotate(img_2, cv2.cv2.ROTATE_90_CLOCKWISE)
                    mask_2=cv2.rotate(mask_2, cv2.cv2.ROTATE_90_CLOCKWISE)
                
                if j%11==0:
                   img_2=cv2.rotate(img_2, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                   mask_2=cv2.rotate(mask_2, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if j%3==0:
                    img_2=cv2.flip(img_2, 1)
                    mask_2=cv2.flip(mask_2, 1)

                if j%6==0:
                    img_2=cv2.flip(img_2, 0)
                    mask_2=cv2.flip(mask_2, 0)

                if img_2.shape[0]==img_2.shape[1] and mask_2.shape[0]==mask_2.shape[1]:
                    cv2.imwrite('./new_t/images/data/'+str(k)+"-"+onlyimg[i]+'.png',img_2)
                    cv2.imwrite('./new_t/masks/data/'+str(k)+"-"+onlyimg[i]+'.png',mask_2)
        

        
        except:
            pass
            #print("error")

        if j==h or time.time()>timeout:
            break
            #if j==h:
             #   h=int(input("Change H?, for no put -1"))
             #   if h==-1:
             #       h=0
             #       break

            #else:
               # break





# In[ ]:





# In[ ]:




