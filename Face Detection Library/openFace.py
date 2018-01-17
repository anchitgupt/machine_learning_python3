# dictionary library
from collections import defaultdict
#main module which work as the classifier for the program
import face_recognition

# os for the reading and writing into the system
# https://docs.python.org/3/library/os.html

# cv2 for image processing

# The shutil module offers a number of high-level operations on files and collections of files like copyfile , copy and copy2
#https://docs.python.org/3/library/shutil.html

import os, cv2, shutil

# it is source folder which contains all unsorted files
# its like test data set for us
unclassified_images_path = "pictures"
unclassified_images_list = os.listdir(unclassified_images_path)

print("Total unclassified images = " + str(len(unclassified_images_list)))

# it is source folder which contains single photographs with their names of photographs
# assume it like train data set

individuals_images_path = "individuals"
individuals_images_list = os.listdir(individuals_images_path)

print("Total individuals = " + str(len(individuals_images_list)))



individuals_name = []
for person in individuals_images_list:
    # jpg images only (can be improved)
    individuals_name.append(person[:-4])

print(individuals_name)
# indivisual name contains all the photos names in indivisual folder

# getting images data i.e. face data in @list -->individuals_face_encodings
# which used to recognition of face


individuals_face_encodings = []
try:
    for individuals in individuals_images_list:
        image = face_recognition.load_image_file(individuals_images_path + "/" + individuals)
        face_encoding = face_recognition.face_encodings(image)[0]
        individuals_face_encodings.append(face_encoding)

except Exception as e:
    print("Error: [" + e.errno + "] " + e.strerr)

# Original dictionary which will contain the classified images(one to many relation) for each individuals
person_picture_collection = defaultdict(lambda: list())


# now reading the @pictures folder each image
for images in unclassified_images_list:


    frame = cv2.imread(unclassified_images_path + "/" + images)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    print(unclassified_images_path + "/" + images)
    # getting face_location in the image
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # encoding it for further use to test
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    #for each face detected it run for one to many relations to detect which face is present
    for face_encoding in face_encodings:
        name = "unknown"
        for index in range(len(individuals_face_encodings)):
            face_distance = face_recognition.face_distance([individuals_face_encodings[index]], face_encoding)
            
            if face_distance < 0.5:
                name = individuals_name[index]
        person_picture_collection[name].append(unclassified_images_path + "/" + images)

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
else:
    shutil.rmtree(results_dir)
    os.makedirs(results_dir)

for person in individuals_name:
    dest = results_dir + "/" + person
    os.makedirs(results_dir + "/" + person)
    for picture in person_picture_collection[person]:
        shutil.copy(picture, dest)
        print("Success")