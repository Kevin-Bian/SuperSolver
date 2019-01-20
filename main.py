import numpy as np
import cv2
import operator
import joblib
from PIL import Image
import numpy as np
import Solver

#classifier
classifier = joblib.load('classifier.pkl')


################################# SETTING UP FOR PROCESSING ########################

# Gather the image and change into grayscale for easier processing
image = cv2.imread('image_path_here')
image_copy = np.copy(image)


def process_img(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian Blur to blur image and remove noise
    # applied on 5x5 pixel grid (this seems good after trial and error)

    blurred_img = cv2.GaussianBlur(grayscale_img, (7, 7), 0)

    # Apply threshold on the image using the mean over a 5x5 window of pixels and subtracts 2 from mean
    #! double check on math/formula for this
    outer_box = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    # Invert colors so lines and numbers are white
    outer_box_img = cv2.bitwise_not(outer_box)

    # dilate image
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3))  # 3x3 kernel used for dilation

    return cv2.dilate(outer_box_img, kernel, iterations=1)


dilated_img = process_img(image_copy)

############################### EXTRACTING GRID ###############################

# Isolate the sudoku grid from the rest of the image using contours
# Idea here is to take the largest external contours because we assume that sudoku grid will be largest (add prompt to make sure)

# This gets all the external contours
new_img, contours, heir = cv2.findContours(
    dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sudoku_grid = sorted(contours, key=cv2.contourArea, reverse=True)[
    0]  # sorts list by area decending and gets largest area


# get max item in a dictionary algo
bottom_left, blank = min(enumerate(
    [pt[0][0] - pt[0][1] for pt in sudoku_grid]), key=operator.itemgetter(1))
top_right, blank = max(enumerate(
    [pt[0][0] - pt[0][1] for pt in sudoku_grid]), key=operator.itemgetter(1))
bottom_right, blank = max(enumerate(
    [pt[0][0] + pt[0][1] for pt in sudoku_grid]), key=operator.itemgetter(1))
top_left, blank = min(enumerate([pt[0][0] + pt[0][1]
                                 for pt in sudoku_grid]), key=operator.itemgetter(1))

sudoku_grid_corners = [
    sudoku_grid[top_left][0],
    sudoku_grid[top_right][0],
    sudoku_grid[bottom_right][0],
    sudoku_grid[bottom_left][0]
]


def scalar_distance(point1, point2):
    return np.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


def crop_and_warp_img(img, rect_coordinates):

    top_left_corner = rect_coordinates[0]
    top_right_corner = rect_coordinates[1]
    bottom_right_corner = rect_coordinates[2]
    bottom_left_corner = rect_coordinates[3]

    # ran into type errors needed to change to float32 as described by opencv docs
    src = np.array([top_left_corner, top_right_corner,
                    bottom_right_corner, bottom_left_corner], dtype='float32')

    # list of distances of coordinates
    corner_distances = [
        scalar_distance(bottom_right_corner, top_right_corner),
        scalar_distance(top_left_corner, bottom_left_corner),
        scalar_distance(bottom_right_corner, bottom_left_corner),
        scalar_distance(top_left_corner, top_right_corner)
    ]

    longest_side = max(corner_distances)

    # this is some linear algebra stuff I had to google how to do this basically making a matrix
    # of our oiginal shape and the shape we want to warp into
    dst = np.array(
        [[0, 0],
         [longest_side - 1, 0],
         [longest_side - 1, longest_side - 1],
         [0, longest_side - 1]],
        dtype='float32')

    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (int(longest_side), int(longest_side)))


cropped_img = crop_and_warp_img(image_copy, sudoku_grid_corners)

dilated_cropped_img = process_img(cropped_img)


############################# Get Gridlines ##########################

# Apply Hough Line Transform, return a list of rho and theta
# get grid


kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

grid = np.zeros(dilated_cropped_img.shape, np.uint8)

#maybe change the votes value for transform for test3
lines = cv2.HoughLines(dilated_cropped_img,1,np.pi/180,300)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(grid,(x1,y1),(x2,y2),(255),2)

grid = cv2.dilate(grid, kernel, iterations=2)


############################# Add Boxes #####################################
font = cv2.FONT_HERSHEY_SIMPLEX #number font
digit_kernel = np.ones((2,2),np.uint8)
_, contours, _ = cv2.findContours(grid,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


digit_list = []
x_coordinates = []
grid_coordinates = []

for cnt in contours[:81]:
    x,y,w,h = cv2.boundingRect(cnt)

    cv2.rectangle(cropped_img,(x,y),(x+w,y+h),(200,0,0),2)

    #grid coordinates
    x_coordinates.append(x)
    grid_coordinates.append([x,y,w,h])
    #adding digits 

    digit=dilated_cropped_img[y:y+h,x:x+w]
    digit = cv2.erode(digit, digit_kernel, iterations=0)
    digit = cv2.resize(digit, (28,28))
    digit = cv2.resize(digit, (28,28))
    
    #check for blanks, if number of pixels are less than 10
    if (np.sum(digit > 0) < 90):
       digit_list.append(0)
       
       

    else:
        num = classifier.predict(np.reshape(digit, (1,-1)))
        digit_list.append(num[0])
        

############################# Extract digits and sorting #############################
def reshape_grid_coordinates(grid_coordinates):
    new_grid_coordinates = [[],[],[],[],[],[],[],[],[]]
    for index in range (9):
        for count in range (9):
            new_grid_coordinates[index].append(grid_coordinates[0])
            del grid_coordinates[0]
    return new_grid_coordinates

def sort_grid_coordinates(grid_coordinates):
    sorted_grid_coordinates = []
    for count in range(9):
        sorted_grid_coordinates.append(sorted(grid_coordinates[count], key=lambda x: int(x[0])))
    return sorted_grid_coordinates

#sort the sudoku grid in order, Sorting digit_array (num values of sudoku) based on x_coordinate_array (x pos of contours)
digit_array = np.reshape(digit_list, (9,9))
x_coordinates_array = np.reshape(x_coordinates, (9,9))
grid_coordinates_array = reshape_grid_coordinates(grid_coordinates)
sorted_grid_coordinates = sort_grid_coordinates(grid_coordinates_array)
sorted_grid =[]
for count in range(8,-1,-1):
    sorted_grid.append([x for blank,x in sorted(zip(x_coordinates_array[count],digit_array[count]))])



################### Putting Digits ##################################
solved_grid = Solver.solve(sorted_grid)

if solved_grid == False:
    print("Please try again, I couldn't process your image or the grid is unsolvable :(")

else: 
    solved_grid.reverse()
    flattened_ans = [item for sublist in solved_grid for item in sublist]
    flattened_grid_coordinates = [item for sublist in sorted_grid_coordinates for item in sublist]



    for count in range(81):
        x = flattened_grid_coordinates[count][0]
        y = flattened_grid_coordinates[count][1]
        w = flattened_grid_coordinates[count][2]
        h = flattened_grid_coordinates[count][3]
        cv2.putText(cropped_img,str(flattened_ans[count]),(x,y+h),font,1,(225,0,0),2)

    # Show image
    cv2.imshow("Picture:", grid)
    cv2.imshow("Picture2:", dilated_cropped_img)
    cv2.imshow("Picture:3", cropped_img)
    cv2.imshow("digit", digit)
    cv2.waitKey(0)


