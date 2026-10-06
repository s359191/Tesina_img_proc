import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision


# ==========================================
# LA TUA FUNZIONE DI DISEGNO ADATTATA
# ==========================================
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


# ==========================================
# INTEGRAZIONE CON IL VIDEO (OpenCV)
# ==========================================

# STEP 2: Create a PoseLandmarker object
# Assicurati di avere il file .task nella stessa cartella
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # IMPORTANTE: diciamo al modello che è un video
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input video (Invece dell'immagine)
video_path = 'videos/pose1.mp4'
cap = cv2.VideoCapture(video_path)

motion_capture_data = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video terminato.")
        break

    # OpenCV legge in BGR, MediaPipe richiede RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Calcoliamo il timestamp del frame in millisecondi (obbligatorio per i video)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # STEP 4: Detect pose landmarks
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    # STEP 5: Process the detection result (Visualizzazione)
    if detection_result.pose_landmarks:
        # Usiamo la tua funzione
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        # Riconvertiamo in BGR per mostrare i colori corretti a schermo
        final_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        landmarks = detection_result.pose_landmarks[0]

        frame_data = []
        for landmark in landmarks:
            # Salviamo x, y, z e la visibilità (quanto il modello è sicuro di quel punto)
            frame_data.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "v": landmark.visibility
            })

        # Aggiungiamo i dati di questo frame alla nostra lista generale
        motion_capture_data.append({
            "time": timestamp_ms,
            "points": frame_data
        })
    else:
        final_frame = frame

    # Mostriamo il risultato con OpenCV
    cv2.imshow("MediaPipe Pose Video", final_frame)

    # Premi 'q' per uscire
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Pulizia
cap.release()
cv2.destroyAllWindows()
detector.close()

with open('animazione_unity.json', 'w') as json_file:
    json.dump(motion_capture_data, json_file, indent=4)

print("Elaborazione finita! Dati salvati in 'animazione_unity.json'")