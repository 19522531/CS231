import cv2 as cv
import numpy as np
import dlib
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
#read img want swap
# act_img1 = cv.imread('static/Bradley.jpg') # ảnh  thay mặt
# act_img2 = cv.imread('static/441px-Jim_Carrey_2008.jpg') # ảnh back ground
#convert to gray img
def swap_face(act_img1, act_img2, predictor):
    act_img1_gray = cv.cvtColor(act_img1,cv.COLOR_BGR2GRAY)
    act_img2_gray = cv.cvtColor(act_img2,cv.COLOR_BGR2GRAY)
    #Create 1 black board like img1
    mask = np.zeros_like(act_img1_gray)
    mask2 = np.zeros_like(act_img2_gray)
    img2_new_face = np.zeros_like(act_img2)
    #Detect face use fuction below
    detector = dlib.get_frontal_face_detector()
    #func help take 68 point in face
    # predictor = dlib.shape_predictor("./__pycache__/shape_predictor_68_face_landmarks.dat")
    #face_1 obtain 2 point create rec cover faces->below:faces_1 obtain 1rec
    faces_1 = detector(act_img1_gray)
    # print(faces_1)
    for face in faces_1:
        #create 68 point
        landmarks = predictor(act_img1_gray,face)
        landmarks_points = []
        for n in range(68):
            #take coor of face(part func dlib)
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x,y))
            # cv.circle(act_img1,(x,y),2,(0,0,255),-1)
        points = np.array(landmarks_points,np.int32)
        #cover face use func cv.convexhull
        convexhull = cv.convexHull(points)
        # cv.polylines(act_img1,[convexhull],True,(0,255,0))
        cv.fillConvexPoly(mask,convexhull,255)
        #and same act_img_1 wwith mask = mask
        face_img_1 = cv.bitwise_and(act_img1,act_img1,mask= mask)
        #create rectrangle cover face
        rect = cv.boundingRect(convexhull)
        #take coor face
        (x,y,w,h) = rect
        #func subdiv2d subdivides a plane into triangles using the Delaunay's algorithm
        subdiv = cv.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        #get point in triangles
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles,np.int32)
        indexes_triangles1 = []
        for t in triangles:
            pt1 = (t[0],t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            #return value type array -> exact index
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles1.append(triangle)
    #process face 2
    faces_2 = detector(act_img2_gray)
    for face in faces_2:

        #create 68 point
        landmarks2 = predictor(act_img2_gray,face)
        landmarks_points2 = []
        for n in range(68):
            #take coor of face(part func dlib)
            x = landmarks2.part(n).x
            y = landmarks2.part(n).y
            landmarks_points2.append((x,y))
            # cv.circle(act_img1,(x,y),2,(0,0,255),-1)
        points_2 = np.array(landmarks_points2,np.int32)
        # #cover face use func cv.convexhull
        convexhull2 = cv.convexHull(points_2)
    for triangle_index in indexes_triangles1:
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        # cv.line(act_img1, tr1_pt1, tr1_pt2, (0, 0, 255))
        # cv.line(act_img1, tr1_pt3, tr1_pt2, (0, 0, 255))
        # cv.line(act_img1, tr1_pt1, tr1_pt3, (0, 0, 255))
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        #Cover triangle
        rect1 = cv.boundingRect(triangle1)
        (x, y, w, h) = rect1
        #crop triangle
        cropped_triangle = act_img1[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        #mapping coor img 1 through crop_rec
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        cv.fillConvexPoly(cropped_tr1_mask, points, 255)
        cropped_triangle = cv.bitwise_and(cropped_triangle, cropped_triangle,
                                        mask=cropped_tr1_mask)
        # Triangulation of second face

        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        # cv.line(act_img2, tr2_pt1, tr2_pt2, (0, 0, 255))
        # cv.line(act_img2, tr2_pt3, tr2_pt2, (0, 0, 255))
        # cv.line(act_img2, tr2_pt1, tr2_pt3, (0, 0, 255))
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_triangle2 = act_img2[y: y + h, x: x + w]
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv.fillConvexPoly(cropped_tr2_mask, points2, 255)
        cropped_triangle2 = cv.bitwise_and(cropped_triangle2, cropped_triangle2,
                                            mask=cropped_tr2_mask)

        #swap triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        #Calc scale 2 rec
        M = cv.getAffineTransform(points,points2)
        #tranform triangle 1 scale ->2
        warped_triangle = cv.warpAffine(cropped_triangle, M, (w, h))

        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area = cv.bitwise_or(img2_new_face_rect_area, warped_triangle,mask=None)
        # img2_new_face_rect_area = cv.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area



    img2_new_face_gray = cv.cvtColor(img2_new_face,cv.COLOR_BGR2GRAY)
    _,background = cv.threshold(img2_new_face_gray,1,255,cv.THRESH_BINARY_INV)
    background = cv.bitwise_and(act_img2,act_img2,mask= background)


    result = cv.bitwise_or(background,img2_new_face)

    img2_face_mask = np.zeros_like(act_img2_gray)
    img2_head_mask = cv.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv.bitwise_not(img2_head_mask)

    (x, y, w, h) = cv.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    dst = result
    kernel = np.ones((3,3), np.float32)/9
    dst = cv.filter2D(dst, -1, kernel)

    # dst = cv.GaussianBlur(result,(5,5),cv.BORDER_DEFAULT)
    img = cv.seamlessClone(dst,act_img2 ,img2_head_mask ,center_face2, cv.NORMAL_CLONE)
    img = cv.medianBlur(img,1)
    return img


# img = swap_face(act_img1, act_img2)
# cv.imshow("img", img)
# cv.waitKey(0)

