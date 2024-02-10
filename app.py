import mediapipe as mp 
import cv2
import time
import numpy as np
import streamlit as st
import simpleaudio as sa
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def check_in_area(point_list, face_list, landmark_list):
    polygon_point_list = [(face_list[index][0],face_list[index][1]) for index in landmark_list]
    poly = Polygon(polygon_point_list)

    point = Point((point_list[8][0],point_list[8][1]))

    return point.within(poly)


def main():

    #Setup page configurations
    st.set_page_config(page_title="Trichotillomania Detection")
    st.title("Trichotillomania Detection via Computer Vision")
    st.caption("Powered by OpenCV, Mediapipe and Streamlit")

    col1, col2 = st.columns(2)

    with col1:
        eyebrow_tracking = st.toggle('Eyebrow Picking Detection')
        beard_tracking = st.toggle('Beard Picking Detection')
        eyelash_tracking = st.toggle('Eyelashes Picking Detection')

    with col2:
        fps_text = st.empty()
        nmr_detections_text = st.empty()
        is_picking_text = st.empty()

    #https://github.com/google/mediapipe/issues/3191
    #https://www.geeksforgeeks.org/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/
    #https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

    #Computer Vision configuration:
        
    #For webcam input (In this case we use the first webcam found):
    cap = cv2.VideoCapture(0)
    #Frame to insert the camera feed
    frame_placeholder = st.empty()

    #Models for drawing landmarks
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic   
    
    #Loading the alert sound from file "bells.wav"
    wave_object = sa.WaveObject.from_wave_file('bells.wav') 

    # Initializing setup variables that will be needed
    previousTime = 0
    currentTime = 0
    number_detections = 0
    first_setup = False
    is_picking = False

    #Make it so that the lists for landmarks on each hand are not empty when checking for picking, otherwise errors will occur 
    zero_list = np.zeros((21,),dtype='i,i').tolist()
    rh_List = zero_list
    lh_List = zero_list

    #Information on facial landmarks found here: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
    eyebrow_region_landmark_index = [336,296,334,293,300,285,295,282,283,276,70,63,105,66,107,56,53,52,65,55]
    eyelash_region_landmark_index = [226,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25,463,414,286,258,257,259,260,467,446,255,339,254,253,252,256,341]
    beard_region_landmark_index = [127,34,116,123,50,205,203,423,425,280,352,454,366,323,401,361,435,288,367,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234]

    #We will use the holistic model from Mediapipe with a confidence of 50%
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, frame = cap.read()

            #Making sure that we can successfully capture the webcam feed
            if success:
                image_height, image_width, _ = frame.shape
            else:
                print("Ignoring empty camera frame.")
                continue

            #Function to convert normalized landmarks into pixel coordinates
            def convert_to_coords(lm):
                xy = [(l.x, l.y) for l in lm]
                return np.multiply(xy, [image_width, image_height]).astype(int)

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #When we run the holistic model on the frame we will get the landmarks for facial and hand features found
            results = holistic.process(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            #Only run the detection if the face is being tracked
            if results.face_landmarks:
                #Get the list for face landmark coordinates
                face_List = convert_to_coords(results.face_landmarks.landmark)

                #Get the list for left and right hand landmark coordinates
                if results.right_hand_landmarks:
                    rh_List = convert_to_coords(results.right_hand_landmarks.landmark)
                else:
                    rh_List = zero_list
                if results.left_hand_landmarks:
                    lh_List = convert_to_coords(results.left_hand_landmarks.landmark)
                else:
                    lh_List = zero_list
                
                #For each type of tracking we check if the index finger of each hand is 
                if eyebrow_tracking:
                    is_eyebrow_picking = check_in_area(lh_List,face_List,eyebrow_region_landmark_index) or check_in_area(rh_List,face_List,eyebrow_region_landmark_index) 
                else: is_eyebrow_picking = False

                if eyelash_tracking:
                    is_eyelash_picking = check_in_area(lh_List,face_List,eyelash_region_landmark_index) or check_in_area(rh_List,face_List,eyelash_region_landmark_index)
                else: is_eyelash_picking = False

                if beard_tracking:
                    is_beard_picking = check_in_area(lh_List,face_List,beard_region_landmark_index) or check_in_area(rh_List,face_List,beard_region_landmark_index)
                else: is_beard_picking = False

                #If any type of tracking is being detected, we conclude that picking is be being done
                is_picking = is_eyebrow_picking or is_eyelash_picking or is_beard_picking
                
            #If Picking is Detected
            if is_picking:
                is_picking_text.markdown(':red[Hair picking detected!]')
                if first_setup == False:
                    play_obj = wave_object.play()
                    first_setup = True
                    number_detections += 1
                else:
                    if not play_obj.is_playing():
                        play_obj = wave_object.play()
                        number_detections += 1
            else:
                if first_setup == True and not play_obj.is_playing():
                    is_picking_text.markdown(':green[No Detection]')

            #Draw Landmarks
            # mp_drawing.draw_landmarks(
            #     frame,
            #     results.face_landmarks,
            #     mp_holistic.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
            # Calculating the FPS
            currentTime = time.time()
            fps = 1 / (currentTime-previousTime)
            previousTime = currentTime

            # Flip the image horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")

            # Text display for fps and number of events 
            fps_text.text("FPS: "+str(int(fps)))
            nmr_detections_text.text("Number of detected events: "+str(number_detections))
            

    # When all the process is done, release the capture
    cap.release()

if __name__ == "__main__":
    main()
