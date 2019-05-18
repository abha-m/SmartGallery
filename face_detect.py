# Reference: https://codereview.stackexchange.com/questions/156736/cropping-faces-from-images-in-a-directory

import cv2
import glob
import os

def facechop(image):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    cropped_face = []

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        # face_file_name = "/home/abha/Celeb_faces/train/all_face_detect_cropped/"  + "_" + str(y) + ".jpg"
        # cv2.imwrite(face_file_name, sub_face)

        cropped_face.append(sub_face)

    # cv2.imshow(image, img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


    return cropped_face

def getCroppedImages(folderpath):
    images = glob.glob(folderpath + "/*")
    file_path_and_cropped_image = {}

    for imgpath in images[0:]:
        cropped = facechop(imgpath)
        if len(cropped) == 1 or len(cropped) == 2:
            cropped_img = cropped[0]
            file_path_and_cropped_image[imgpath] = cropped_img

    return file_path_and_cropped_image


if __name__ == '__main__':

    a = getCroppedImages('/home/abha/Celeb_faces/train/all')
    print(a['/home/abha/Celeb_faces/train/all/httpgonetworthcomwpcontentuploadsthumbsjpg.jpg'])