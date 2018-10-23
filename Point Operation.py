#!/usr/bin/env python
# coding: utf-8

# # Point Operation

# ## Modifying Image Intensity

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

imBGR = cv2.imread('p1.jpg')
imRGB = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB) # change image pattern into RGB
imGRAY = cv2.cvtColor(imRGB, cv2.COLOR_RGB2GRAY) # change image pattern into Gray scale

imRGB_int16 = np.int16(imRGB) # extend values range by changing a data type into int-16-bit for further operation
imGRAY_int16 = np.int16(imGRAY)

plt.imshow(imRGB),plt.title("Original Image")
#plt.imshow(imGRAY,'gray')
plt.show()


# ### Contrast Adjustment (with limiting the result by clamping)

# In[2]:


contr_im = imRGB_int16 * 1.5 # increase contrast by factor = 1.5
contr_im_clip = np.clip(contr_im, 0, 255) # limit values in range (0,255)
contr_im_uint8 = np.uint8(contr_im_clip) # change a data type into unsigned-int-8-bit for image show
#p1_rgb_contr = plt.imshow(contr_im_uint8)

plt.subplot(2,2,1),plt.imshow(imRGB)
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(contr_im_uint8)
plt.title("Contrast Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Contrast Image Histogram 

# In[3]:


plt.subplot(3,1,1),plt.hist(imRGB.ravel(),256,[0,256])
plt.title("Original Image")#, plt.xticks([]),plt.yticks([])
plt.subplot(3,1,3),plt.hist(contr_im_uint8.ravel(),256,[0,256])
plt.title("Contrast Image")#, plt.xticks([]),plt.yticks([])
plt.show()


# ### Brightness Adjustment (with limiting the result by clamping)

# In[4]:


bright_im = imRGB_int16 + 20 # increase brightness by adding a value
bright_im_clip = np.clip(bright_im, 0, 255) # limit values in range (0,255)
bright_im_uint8 = np.uint8(bright_im_clip)
#p1_rgb_bright = plt.imshow(bright_im_uint8)

plt.subplot(2,2,1),plt.imshow(imRGB)
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(bright_im_uint8)
plt.title("Brightness Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Brightness Image Histogram

# In[5]:


plt.subplot(3,1,1),plt.hist(imRGB.ravel(),256,[0,256])
plt.title("Original Image")#, plt.xticks([]),plt.yticks([])
plt.subplot(3,1,3),plt.hist(bright_im_uint8.ravel(),256,[0,256])
plt.title("Brightness Image")#, plt.xticks([]),plt.yticks([])
plt.show()


# ### Inverting Image

# In[6]:


inv_im = -imRGB + 255 # reverse the order of pixel values
#p1_rgb_inv = plt.imshow(inv_im)

plt.subplot(2,2,1),plt.imshow(imRGB)
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(inv_im)
plt.title("Inverting Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Inverting Image Histogram

# In[7]:


plt.subplot(3,1,1),plt.hist(imRGB.ravel(),256,[0,256])
plt.title("Original Image")#, plt.xticks([]),plt.yticks([])
plt.subplot(3,1,3),plt.hist(inv_im.ravel(),256,[0,256])
plt.title("Inverting Image")#, plt.xticks([]),plt.yticks([])
plt.show()


# ### Threshold Operation

# In[8]:


ret, th_im = cv2.threshold(imGRAY,60,255,cv2.THRESH_BINARY) # modify image with threshold value = 60
#p1_gray_th = plt.imshow(th_im, 'gray')

plt.subplot(2,2,1),plt.imshow(imGRAY, 'gray')
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(th_im, 'gray')
plt.title("Threshold Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Threshold Image Histogram

# In[9]:


plt.subplot(3,1,1),plt.hist(imGRAY.ravel(),256,[0,256])
plt.title("Original Image"), #plt.xticks([]),plt.yticks([])
plt.subplot(3,1,3),plt.hist(th_im.ravel(),256,[0,256])
plt.title("Threshold Image"), #plt.xticks([]),plt.yticks([])
plt.show()


# ### Auto-contrast

# In[10]:


max_val = 0
min_val = 255

for i in range(0, imGRAY.shape[0]):
    if max(imGRAY[i]) > max_val:
        max_val = max(imGRAY[i])
print(max_val)

for j in range(0, imGRAY.shape[0]):
    if min(imGRAY[j]) < min_val:
        min_val = min(imGRAY[j])
print(min_val)


# In[11]:


# plot histogram of the original gray scale image
plt.hist(imGRAY.ravel(),256,[0,256]), plt.title("Original Image")
plt.show()


# In[12]:


# from the image histogram of 8-bit gray image(amin = 0, amax = 255), define alow =0 and ahigh = 220
alow = 0; ahigh = 220; amin = 0; amax = 255
auto_contr_factor = (amax-amin)/(ahigh-alow) 
auto_contr_im = amin + ((imGRAY_int16 - alow) * auto_contr_factor)
auto_contr_im_clip = np.clip(auto_contr_im, 0, 255)
auto_contr_im_uint8 = np.uint8(auto_contr_im_clip)
#p1_gray_auto_contr = plt.imshow(auto_contr_im_uint8)

plt.subplot(2,2,1),plt.imshow(imGRAY, 'gray')
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(auto_contr_im_uint8, 'gray')
plt.title("Auto-Contrast Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Auto-Contrast Histogram

# In[13]:


plt.subplot(3,1,1),plt.hist(imGRAY.ravel(),256,[0,256])
plt.title("Original Image"), #plt.xticks([]),plt.yticks([])
plt.subplot(3,1,3),plt.hist(auto_contr_im_uint8.ravel(),256,[0,256])
plt.title("Auto-Contrast Image"), #plt.xticks([]),plt.yticks([])
plt.show()


# ### Modified Auto-contrast

# In[14]:


# find a histogram of the gray scale image
hist,bins = np.histogram(imGRAY.ravel(),256,[0,256])


# In[15]:


#find new alow and ahigh values by saturating both end of image histogram 0.5%
def lowhigh_vals():
    ql = 0
    for i in range(len(hist)):
        ql = ql + hist[i]
        if ql >= 0.005*sum(hist):
            break
            
    qh = sum(hist)
    for j in range(len(hist)):
        qh = qh - hist[j]
        if qh <= 0.005*sum(hist):
            break
    return i,j
print("(alown, ahighn) = ",lowhigh_vals())


# In[16]:


# using alown and ahighn from a function: lowhigh_vals() 
alown = 4; ahighn = 223; amin = 0; amax = 255
Mod_auto_contr_factor = (amax-amin)/(ahighn-alown) 
Mod_auto_contr_im = amin + ((imGRAY_int16 - alown) * auto_contr_factor)

Mod_auto_contr_im_clip = np.clip(Mod_auto_contr_im, 0, 255)
Mod_auto_contr_im_uint8 = np.uint8(Mod_auto_contr_im_clip)
#p1_gray_Mod_auto_contr = plt.imshow(Mod_auto_contr_im_uint8)

plt.subplot(2,2,1),plt.imshow(imGRAY, 'gray')
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(Mod_auto_contr_im_uint8, 'gray')
plt.title("Modified Auto-Contrast Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Modified Auto-Contrast Histogram

# In[17]:


plt.subplot(3,1,1),plt.hist(imGRAY.ravel(),256,[0,256])
plt.title("Original Image"), #plt.xticks([]),plt.yticks([])
plt.subplot(3,1,3),plt.hist(Mod_auto_contr_im_uint8.ravel(),256,[0,256])
plt.title("Modified Auto-Contrast Image"), #plt.xticks([]),plt.yticks([])
plt.show()


# ### Histogram Equalization

# In[18]:


hist_imGRAY,bins = np.histogram(imGRAY.flatten(),256,[0,256])
cum_hist = hist_imGRAY.cumsum() # cumulative histogram
cum_hist_normalized = cum_hist * hist_imGRAY.max()/ cum_hist.max() # cumulative histogram normalization

plt.hist(imGRAY.flatten(),256,[0,256])
plt.plot(cum_hist_normalized, color = 'g')
plt.xlim([0,256])
plt.legend(('image histogram','cdf'), loc = 'upper left')
plt.show()


# In[19]:


# Histogram Equalization Function(hist_eq)
hist_eq = cum_hist*255/cum_hist.max() # cum_hist.max() is an image size (M*N)
hist_eq_uint8 = np.uint8(hist_eq)


# In[20]:


imGRAY_eq = hist_eq_uint8[imGRAY]

hist_imGRAY_eq,bins = np.histogram(imGRAY_eq.flatten(),256,[0,256])
cum_hist_eq = hist_imGRAY_eq.cumsum()
cum_hist_eq_normalized = cum_hist_eq * hist_imGRAY_eq.max()/ cum_hist_eq.max()

plt.plot(cum_hist_eq_normalized, color = 'g')
plt.hist(imGRAY_eq.flatten(),256,[0,256])
plt.xlim([0,256])
plt.legend(('equalized histogram','cdf'), loc = 'upper left')
plt.show()


# In[21]:


plt.subplot(2,2,1),plt.imshow(imGRAY, 'gray')
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(imGRAY_eq, 'gray')
plt.title("Equalized Image"), plt.xticks([]),plt.yticks([])
plt.show()


# ### Histogram Specification

# In[22]:


hist_imGRAY,bins = np.histogram(imGRAY.flatten(),256,[0,256])
cdf = hist_imGRAY.cumsum() 
cdf_normalized = cdf / cdf.max() 

plt.plot(cdf_normalized, color = 'r')
plt.xlim([0,256])
plt.legend(('cdf',), loc = 'upper left')
plt.show()


# In[23]:


# Piecewise Linear Distribution

a = [0,51,102,153,204,255]
PL = cdf_normalized

for i in range(0,256):
    if a[0] <= i < a[1]:
        PL[i] = PL[a[0]] + ((i-a[0])*(PL[a[1]] - PL[a[0]])/(a[1]-a[0]))
    elif a[1] <= i < a[2]:
        PL[i] = PL[a[1]] + ((i-a[1])*(PL[a[2]] - PL[a[1]])/(a[2]-a[1])) 
    elif a[2] <= i < a[3]:
        PL[i] = PL[a[2]] + ((i-a[2])*(PL[a[3]] - PL[a[2]])/(a[3]-a[2]))    
    elif a[3] <= i < a[4]:
        PL[i] = PL[a[3]] + ((i-a[3])*(PL[a[4]] - PL[a[3]])/(a[4]-a[3]))
    elif a[4] <= i < a[5]:
        PL[i] = PL[a[4]] + ((i-a[4])*(PL[a[5]] - PL[a[4]])/(a[5]-a[4]))
    else:
        PL[i] = 1


# In[24]:


plt.plot(PL, color = 'b')
plt.plot(cdf_normalized, color = 'r')
plt.xlim([0,256])
plt.legend(('Piecewise Linear Distribution','cdf'), loc = 'upper left')
plt.show()


# In[25]:


imGRAY_sp = PL[imGRAY]


# In[26]:


plt.subplot(2,2,1),plt.imshow(imGRAY, 'gray')
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(imGRAY_sp, 'gray')
plt.title("Specified Image"), plt.xticks([]),plt.yticks([])
plt.show()

