import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def display_image(image): # function to display image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def save_image(savename, image):
    cv2.imwrite(savename , image)


def convert_to_uint8(matrix):
    matrix = np.array(matrix, dtype=np.uint8)
    return matrix


def reduce_resolution(image, y1,y2): # function to reduce size of the image

    dim = (y2, y1)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #print('Resized Dimensions : ', resized.shape)
    return resized


def cal_hist(img, channel): # funtion to calcullate the histogram of the input image
    histg = cv2.calcHist([img], [channel], None, [256], [0, 190])
    return histg


def avgnoise_reduce(image, filtersize): #smoothes the image to get rid fo the noise by using average filter
    avgkernel = np.ones((filtersize, filtersize), np.float32) / (filtersize * filtersize)
    noisereduced = cv2.filter2D(image, -1, avgkernel)
    return noisereduced



def similarity_coeff(image_hist, template_hist): # this function calculates the measure of similarity between two histograms
    sum = 0
    diff = image_hist - template_hist
    diff = np.square(diff)
    sum = np.sum(diff)
    sum = 1 + sum
    sum = 1 / sum

    # print(sum)
    return sum

def padimage(image,window_size): # funtion to padd image
    x = np.pad(image , pad_width=int((window_size)/2), mode='constant', constant_values=-1)
    return x

def jc(result, reference):
    """
    Jaccard coefficient

    Computes the Jaccard coefficient between the binary objects in two images.

    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc




def covariance_matrix(image, template, b_hist,g_hist, r_hist):  # function to slide across the image and compare with the template , this will return the covariance matrix and the threshold
    cov_matrix = np.zeros((image.shape[0] , image.shape[1] ))
    b_image, g_image, r_image = cv2.split(image)
    b_image=padimage(b_image,120)
    g_image=padimage(g_image,120)
    r_image=padimage(r_image,120)
    b_template, g_template, r_template = cv2.split(template)
    threshold = -100

    b_template_hist = cal_hist(b_template, 0)
    g_template_hist = cal_hist(g_template, 0)
    r_template_hist = cal_hist(r_template, 0)

    for x in range(image.shape[0] ):
        for y in range(image.shape[1] ):

            b_patch = cal_hist(b_image[x:x + 120, y:y + 120], 0)
            g_patch = cal_hist(g_image[x:x + 120, y:y + 120], 0)
            r_patch = cal_hist(r_image[x:x + 120, y:y + 120], 0)



            b_cov = similarity_coeff(b_patch, b_hist)
            g_cov = similarity_coeff(g_patch, g_hist)
            r_cov = similarity_coeff(r_patch, r_hist)


            cov = (b_cov * 1) + (g_cov * 2) + (r_cov * 0.5)

            cov_matrix[x, y] = cov

            if (cov > threshold):
                threshold = cov

    threshold = 0.5 * threshold
    #print("threshold", threshold)

    return cov_matrix, threshold


img1 = cv2.imread('IDRiD_25.jpg', 1)     #### loading to obtain templates
img2 = cv2.imread('IDRiD_01_full.jpg', 1)
img3 = cv2.imread('IDRiD_02_full.jpg', 1)
img4 = cv2.imread('IDRiD_04.jpg', 1)
img5 = cv2.imread('IDRiD_49.jpg', 1)


img1 = reduce_resolution(img1, 569,857)  ## reducing the size of the images
img2 = reduce_resolution(img2, 569,857)
img3 = reduce_resolution(img3, 569,857)
img4 = reduce_resolution(img4, 569,857)
img5 = reduce_resolution(img5, 569,857)
#print(img1)
#print("image shape",img.shape)
window = 120

noisereduced1 = avgnoise_reduce(img1, 6)   ## avg filter applied to reduce the noise in the imagees
noisereduced1 = convert_to_uint8(noisereduced1)
noisereduced2 = avgnoise_reduce(img2, 6)
noisereduced2 = convert_to_uint8(noisereduced2)
noisereduced3 = avgnoise_reduce(img3, 6)
noisereduced3 = convert_to_uint8(noisereduced3)
noisereduced4 = avgnoise_reduce(img4, 6)
noisereduced4 = convert_to_uint8(noisereduced4)
noisereduced5 = avgnoise_reduce(img5, 6)
noisereduced5 = convert_to_uint8(noisereduced5)

#print(noisereduced4.shape)
save_image("noise_reduced.jpg", noisereduced5)

# gray_image = cv2.cvtColor(noisereduced3, cv2.COLOR_BGR2GRAY)


# v=0

#display_image(noisereduced1)

colourblot1 = noisereduced1[220:220 + 120, 160:160 + 120]  ## manually cropping the optic disc of  four images used to get obtain template.
#display_image(colourblot1)
colourblot2 = noisereduced2[220:220 + 120, 580:580 + 120]
colourblot3 = noisereduced3[170:170 + 120, 80:80 + 120]
colourblot4 = noisereduced4[90:90 + 120, 330:330 + 120]
#display_image(colourblot4)
b_template1, g_template1, r_template1 = cv2.split(colourblot1)
b_template2, g_template2, r_template2 = cv2.split(colourblot2)
b_template3, g_template3, r_template3 = cv2.split(colourblot3)
b_template4, g_template4, r_template4 = cv2.split(colourblot4)

#print(r)
print("colour blot", colourblot1)
print("colour blot shape", colourblot1.shape)

#b_image, g_image, r_image = cv2.split(noisereduced3)

b_template_hist1 = cal_hist(b_template1, 0)
#plt.hist(b_template1.ravel(),256,[0,256])
#plt.show()
g_template_hist1 = cal_hist(g_template1, 0)
r_template_hist1 = cal_hist(r_template1, 0)
#plt.hist(r_template1.ravel(),256,[0,256])
#plt.show()

b_template_hist2 = cal_hist(b_template2, 0)
g_template_hist2 = cal_hist(g_template2, 0)
r_template_hist2 = cal_hist(r_template2, 0)

b_template_hist3 = cal_hist(b_template3, 0)
g_template_hist3 = cal_hist(g_template3, 0)
r_template_hist3 = cal_hist(r_template3, 0)

b_template_hist4 = cal_hist(b_template4, 0)
g_template_hist4 = cal_hist(g_template4, 0)
r_template_hist4 = cal_hist(r_template4, 0)

b_template_hist = (b_template_hist1 + b_template_hist2 + b_template_hist3 + b_template_hist4) / 4   ##### avergaing the four histograms to obtain tamplate
g_template_hist = (g_template_hist1 + g_template_hist2 + g_template_hist3 + g_template_hist4) / 4
r_template_hist = (r_template_hist1 + r_template_hist2 + r_template_hist3 + r_template_hist4) / 4

#plt.hist(g_template_hist.ravel(),256,[0,200])
#plt.show()

# print("histogram",r_template_hist)
# print("hist shape",r_template_hist.shape)
# threshold=-10000

c_matrix = np.zeros((noisereduced1.shape[0], noisereduced1.shape[1] ))
print("c_matrix shape", c_matrix.shape)



print("noise reduced shape",noisereduced5.shape)

path='/Users/allen/Desktop/computer vision/cvproject/Data_Individual_Component/original_retinal_images'
midpath='/Users/allen/Desktop/computer vision/cvproject/Data_Individual_Component'
x=os.listdir(path)
x.sort()
x=x[1:]
#print("xxxxx",x)

y=os.listdir(midpath+'/optic_disc_segmentation_masks')
y.sort()
#print("yyyyy",y)



#for r,d,f in os.walk(path):
    #print(f)
count=1
matchvalue=[]
for i in range(len(x)) :  # looping to get evey image in our dataset
    file=x[i]
    ground_file=y[i]

    #print(file)

    imgpath='/Users/allen/Desktop/computer vision/cvproject/Data_Individual_Component/original_retinal_images/'+ file
    groundpath=midpath+'/optic_disc_segmentation_masks/'+ground_file
    #resultpath=
    #print(imgpath)

    ground_file=cv2.imread(groundpath,cv2.IMREAD_GRAYSCALE)
    ground_file = reduce_resolution(ground_file, 569, 857)

    img_main=cv2.imread(imgpath,1)
    img_main=reduce_resolution(img_main, 569,857)
    binary_matrix = np.zeros((img_main.shape[0], img_main.shape[1]))
    binary_output=convert_to_uint8(binary_matrix)
    img_main=avgnoise_reduce(img_main, 6)
    c_matrix,threshold=covariance_matrix(img_main,colourblot1,b_template_hist,g_template_hist,r_template_hist)  ## will return the covariance matrix between each image and template.
    retval, binary_image=cv2.threshold(c_matrix,threshold,255,cv2.THRESH_BINARY)
    M = cv2.moments(binary_image)

    #calculate x,y coordinate of cente
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    #print("x cordinate of the image", cX)
    #print("y corfinate of the image", cY)

    cv2.circle(binary_output, (cX, cY), int((window-20) / 2), (255, 0, 0), -1)  #draws a solid circle with centre cx and cy of radius (window-20/2)
    #cv2.imshow('Input', img_in.astype(np.uint8))
    #cv2.imshow('Output', image_main.astype(np.uint8))
    #display_image(img_main)

    jccoef=jc(binary_output,ground_file) # calculates jc coeef of the inputs
    matchvalue.append(jccoef)

    #save_image(os.path.join(midpath+'/markedresults/' +'OG_'+file), binary_output )

    print("jc for image", count,'=',jccoef)
    count=count+1


print('average of the list',(sum(matchvalue) / len(matchvalue) ))







