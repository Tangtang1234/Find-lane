import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""

    return cv2.Canny(img, low_threshold, high_threshold)

def sobel(img):
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    edges1 = cv2.filter2D(img, -1, sobel_y)
    edges = cv2.filter2D(edges1, -1, sobel_x)
    return edges

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #vertices = np.array([[(50, 530), (450, 320), (1000, 530), (500, 320)]], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 255, 0], thickness=20):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #通过极点坐标画线
    leftx=[]
    lefty=[]
    rightx =[]
    righty =[]
    middle = 30
    imshape = img.shape
    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            if((x1 < int(imshape[1] * 0.45) + middle)  & ( y1 > int(imshape[0]) * 0.6) ) :
                if (x1 >180):
                    leftx.append(x1)
                    lefty.append(y1)
            if (( x2 < int(imshape[1] * 0.45 + middle)) & ( y2 > int(imshape[0]) * 0.6)):
                if (x2 > 180):
                    leftx.append(x2)
                    lefty.append(y2)

            if ( x1 > int(imshape[1] * 0.6) - middle) & ( y1 > int(imshape[0] * 0.6)):
                rightx.append(x1)
                righty.append(y1)
            if (x2 > int(imshape[1] * 0.6) - middle)  &  (y2 > int(imshape[0] * 0.6)):
                rightx.append(x2)
                righty.append(y2)

    #print (leftx,lefty,rightx,righty)
    #left_xy = [( max(leftx),min(lefty)),(min(leftx),max[lefty] ) ]
    #left_line
    cv2.line(img,( max(leftx),min(lefty)), (min(leftx),max(lefty)),color,thickness)
    #Right_lane
    cv2.line(img,(min(rightx), min(righty)), (max(rightx), max(righty)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def find_lane(img):
    imshape = img.shape
     #相机作用范围
     #vertices = np.array([[(0, imshape[0]), (0, 0), (imshape[1], 0), (imshape[1], imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0, imshape[0]), (int(0.45 * imshape[1]), int(0.65 * imshape[0])),
                          (int(0.6 * imshape[1]), int(0.65 * imshape[0])), (imshape[1], imshape[0])]], dtype=np.int32)

     #vertices = np.array([[(100, 540), (435, 335),(515, 335),(960,540)]], dtype=np.int32)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
     #line_image = np.copy(image) * 0  # creating a blank to draw lines on

    gray_image = grayscale(img)
    gauss_imgae = gaussian_blur(gray_image,5)
    #plt.imshow(gauss_imgae)
    canny_image = canny(gauss_imgae,30,150)
    #plt.imshow(canny_image)
    sobel_image = sobel(canny_image)
    mask_image = region_of_interest(sobel_image,vertices)
    line_image =  hough_lines(mask_image, rho, theta, threshold, min_line_length, max_line_gap)
    out_image = weighted_img(img,line_image,α=1.0, β=0.8, γ=0.5)
    return out_image


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    standard_im = cv2.resize(image, (960, 540))

    result = find_lane(standard_im)

    return result


images2 = glob.glob('CarND-Advanced-Lane-Lines-master/test_images/*.jpg')


for img in images2:
    image = mpimg.imread(img)

    #img = mpimg.imread ('../test_images/straight_lines2.jpg')
    result_out = process_image(image)
    plt.imshow(result_out)
    plt.show()