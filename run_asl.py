import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from gtts import gTTS
import os
import time

# Load TFLite model
interpreter = tflite.Interpreter(model_path="asl_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

current_word = ""
last_added = 0
COOLDOWN = 0.5

cap = cv2.VideoCapture(0)
print("✅ Camera started. Press q to quit, s to speak word, c to clear word.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img.astype(np.float32)/255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(output)
    pred_label = labels[pred_idx]

    now = time.time()
    if now - last_added > COOLDOWN:
        if pred_label == "SPACE":
            current_word += " "
            last_added = now
        elif pred_label == "DEL" and current_word:
            current_word = current_word[:-1]
            last_added = now
        elif pred_label not in ["NOTHING", "SPACE", "DEL"]:
            current_word += pred_label
            last_added = now

    cv2.putText(frame, f"Pred: {pred_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Word: {current_word}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if current_word.strip():
            tts = gTTS(current_word, lang='en')
            tts.save("output.mp3")
            os.system("mpg123 -q output.mp3")
    elif key == ord('c'):
        current_word = ""

cap.release()
cv2.destroyAllWindows()
