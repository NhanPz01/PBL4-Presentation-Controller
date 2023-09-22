#pip install tensorflow opencv-python mediapipe sklearn matplotlib scikit-learn
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
#NHỚ BẤM Q VÀO CHƯƠNG TRÌNH ĐỂ TẮT

DATA_PATH = os.path.join('MP_Data')

actions = np.array(['Forward','Backward','Present', 'Quit'])

#30 video chứa data
no_sequences = 30

#video sẽ mang 30 frames
sequence_length = 30

#Tạo thư mục data tương ứng với actions
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
        except:
            pass
        


#Holistic Model nhận diện đối tượng và drawing util vẽ ra các line tương ứng 
mp_holistic = mp.solutions.holistic #Holistic model https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#https://www.youtube.com/watch?v=doDUihpj6ro&t=422s 15:40 bắt đầu giải thích, 28:50 có demo xem chuyển màu
#tóm tắt là chuyển màu bình thường về một dạng màu khác (ko rõ có phải là trắng đen hay ko) để tiết kiệm bộ nhớ 
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Đổi màu lượt đầu
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Đổi màu lại như cũ
    return image, results

# vẽ các line lên màn hình
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) #Vẽ các điểm nối mặt , có thể dùng FACEMESH_TESSELATION thay thế
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #Vẽ các điểm nối dáng
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Vẽ các điểm tay trái
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Vẽ các điểm tay phải
# Như trên nhưng đổi màu và độ dày các đường vẽ
def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) #Vẽ các điểm nối mặt , có thể dùng FACEMESH_TESSELATION thay thế
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) #Vẽ các điểm nối dáng
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) #Vẽ các điểm tay trái
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #Vẽ các điểm tay phải
    
#Lấy ra các điểm nối https://www.youtube.com/watch?v=doDUihpj6ro&t=422s 40:00 có giải thích
def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])



print("1.Collect Data")
print("2.Mở camera")
fun = input("Vui lòng chọn việc bạn muốn làm")
if fun == "1":
    #Collect Data qua camera (Quét từng frame lấy các điểm rồi lưu nó thành 1 numpy array)
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence= 0.5,min_tracking_confidence= 0.5) as holistic:
        for action in actions:
            for sequence in range (no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read() #frame là hình ảnh lấy được từ camera
                    #Bắt đầu nhận diện
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)
                    #Vẽ các đường nối
                    draw_styled_landmarks(image, results)
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} video number {}'.format(action,sequence), (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),4,cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} video number {}'.format(action,sequence), (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),4,cv2.LINE_AA)
                    keypoints = extract_keypoints(results)
                    cv2.imshow('OpenCV Feed', image)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    # Tắt camera bằng nút q
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()
    #####

else:
    # Show màn hình
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    #Đặt mediapipe model https://www.youtube.com/watch?v=doDUihpj6ro&t=422s 21:00 có giải thích
    #có thể chỉnh hai thông số cho phù hợp
    with mp_holistic.Holistic(min_detection_confidence= 0.5,min_tracking_confidence= 0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read() #frame là hình ảnh lấy được từ camera
            
            #Bắt đầu nhận diện
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            #Vẽ các đường nối
            draw_styled_landmarks(image, results)
            
            cv2.imshow('OpenCV Feed', image)
            # Tắt camera bằng nút q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    #####
    
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X = np.array(sequences)
y= to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)