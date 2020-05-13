#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils


# In[3]:


img0 = cv2.imread("C:\\Users\\Keshav Rao\\Desktop\\Keshav\\Kaam\\cpp_tasks\\nlp+ocr\\document_adv.png")


# Convert to grayscale

# In[4]:


# gray method 1
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
cv2.imshow("Converted image:", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# Denoise

# In[5]:


img0 = cv2.GaussianBlur(gray, (5,5),0)

plt.subplot(121), plt.imshow(gray), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img0), plt.title('averaging')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.imshow("gfilter", img0)


# In[6]:


img1 = cv2.bilateralFilter(gray, 11, 17, 17)

plt.subplot(121), plt.imshow(gray), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img1), plt.title('averaging')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.imshow("bfilter", img1)


# In[ ]:





# find edges

# In[7]:


''' 
edges = cv2.Canny(gray,1,200)

plt.subplot(121),plt.imshow(gray ,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
''' 


# In[8]:


edges0 = cv2.Canny(img0,0,150)

plt.subplot(121),plt.imshow(img0 ,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges0,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.imshow(".", edges0)


# In[9]:


edges1 = cv2.Canny(img1,1,200)

plt.subplot(121),plt.imshow(img1 ,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges1,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.imshow(".r", edges1)


# In[ ]:





# In[ ]:





# morphological trans

# In[10]:


kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(edges1,kernel,iterations = 1)

cv2.imshow("erosion1", erosion) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges0,kernel,iterations = 1)

cv2.imshow("dilation0", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()


''' 
erosion = cv2.erode(dilation,kernel,iterations = 1)
cv2.imshow(".", erosion) 
''' 


# In[12]:


closing = cv2.morphologyEx(edges0, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing0", closing) 
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("grad dilation", gradient) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


''' 
dilation0 
closing0
''' 
dil = dilation.copy()
clo = closing.copy()
gray1 = gray.copy()


# In[ ]:





# In[ ]:





# Contours

# In[14]:


ret,thresh = cv2.threshold(dilation.copy(),127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# In[ ]:


type(contours)


# In[ ]:


largest_area = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>largest_area:
        cntf = cnt
        largest_area = area
        #largest_contour_index=i
        x,y,w,h = cv2.boundingRect(cnt)

#cnts = imutils.grab_contours(contours)
c = max(contours, key=cv2.contourArea)
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv2.drawContours(gray1, [c], -1, (0, 255, 255), 2)
cv2.imshow(".", gray1)
cv2.waitKey(0)


# In[ ]:


dila = cv2.rectangle(dilation.copy(),(x,y),(x+w,y+h),(0,255,0),2)    
dila = cv2.rectangle(gray.copy(), (x,y), (x+w,y+h), (0,255,0), 2)
    
cv2.imshow("Result", dila)

cv2.waitKey(0)


# In[ ]:


dila1 = dila.copy()
cv2.imshow(".", dila1)
cv2.waitKey(0)


# perspective wrap

# In[ ]:





# In[ ]:


# Binarization

th = cv2.adaptiveThreshold(edged.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,11,2)
cv2.imshow("Converted image:", th)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 
# 

# In[ ]:





# In[ ]:





# 
