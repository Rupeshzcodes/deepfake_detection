import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image, ExifTags
import io

class ForensicAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5, # Uplifted for Multi-face
            refine_landmarks=True, 
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load Haar Cascade for Glasses Detection
        self.glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    def _get_coords(self, landmark, w, h):
        return np.array([int(landmark.x * w), int(landmark.y * h)])

    def _crop_roi(self, image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return cv2.mean(image, mask=mask)[:3] 

    # -------------------------------------------------------------------------
    # MODULE A: Bio-Metric Eye Scanner
    # -------------------------------------------------------------------------
    def check_eye_biometrics(self, image, landmarks, w, h):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            left_pupil = self._get_coords(landmarks[468], w, h)
            right_pupil = self._get_coords(landmarks[473], w, h)
            
            def analyze_eye(pupil_center, roi_size=20):
                x, y = pupil_center
                x1, y1 = max(0, x-roi_size), max(0, y-roi_size)
                x2, y2 = min(w, x+roi_size), min(h, y+roi_size)
                roi = gray[y1:y2, x1:x2]
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(roi)
                return (maxLoc[0] - roi_size, maxLoc[1] - roi_size)
            
            l_glint = analyze_eye(left_pupil)
            r_glint = analyze_eye(right_pupil)
            
            # 1. Glint Consistency
            dist = np.linalg.norm(np.array(l_glint) - np.array(r_glint))
            
            # 2. EAR (verticality)
            l_top = self._get_coords(landmarks[159], w, h)
            l_bot = self._get_coords(landmarks[145], w, h)
            r_top = self._get_coords(landmarks[386], w, h)
            r_bot = self._get_coords(landmarks[374], w, h)
            
            l_h = np.linalg.norm(l_top - l_bot)
            r_h = np.linalg.norm(r_top - r_bot)
            
            score = 1.0
            msg = "Bio-metrics Verified"
            
            if dist > 5.0:
                score -= 0.6
                msg = "Glints Mismatched (Dead Eyes)"
            
            if abs(l_h - r_h) > 5.0:
                 score -= 0.2
                 msg += " & Asymmetric Eye Opening"

            score = max(0.0, score)
            return {"score": score, "details": f"{msg} (Glint Dist: {dist:.1f})"}
        except Exception:
            return {"score": 0.5, "details": "Bio-metric Scan Failed"}

    # -------------------------------------------------------------------------
    # MODULE B: Frequency Domain Analysis (FFT)
    # -------------------------------------------------------------------------
    def check_frequency(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
            
            rows, cols = gray.shape
            crow, ccol = rows//2, cols//2
            mask_size = 30
            magnitude_spectrum[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
            
            high_freq_energy = np.mean(magnitude_spectrum)
            score = min(high_freq_energy / 150.0, 1.0)
            
            return {"score": score, "spectrum": magnitude_spectrum, "details": f"High Freq Energy: {high_freq_energy:.1f}"}
        except Exception as e:
            return {"score": 0.5, "spectrum": np.zeros((100,100)), "details": str(e)}

    # -------------------------------------------------------------------------
    # MODULE C: Pixel & Compression Artifacts (ELA)
    # -------------------------------------------------------------------------
    def check_ela(self, image):
        try:
            _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resaved = cv2.imdecode(buffer, 1)
            diff = cv2.absdiff(image, resaved)
            ela_image = cv2.convertScaleAbs(diff, alpha=10.0, beta=0)
            mean_diff = np.mean(ela_image)
            score = min(mean_diff / 50.0, 1.0)
            return {"score": score, "ela_image": ela_image, "details": f"ELA Intensity: {mean_diff:.1f}"}
        except Exception as e:
            return {"score": 0.5, "ela_image": np.zeros_like(image), "details": str(e)}

    # -------------------------------------------------------------------------
    # MODULE D: Head Pose
    # -------------------------------------------------------------------------
    def check_head_pose(self, image, landmarks, w, h):
         try:
             face_2d = []
             face_3d = []
             # Nose, Chin, Left Eye, Right Eye, Mouth L, Mouth R
             key_defs = [(1, [0, -1.12, 7.47]), (199, [0, -17.04, 0]), 
                         (33, [-22.5, 7.47, 0]), (263, [22.5, 7.47, 0]), 
                         (61, [-15.0, -8.72, 0]), (291, [15.0, -8.72, 0])]
             
             for idx, coords in key_defs:
                 face_2d.append([landmarks[idx].x * w, landmarks[idx].y * h])
                 face_3d.append(coords)
             
             face_2d = np.array(face_2d, dtype=np.float64)
             face_3d = np.array(face_3d, dtype=np.float64)
             
             focal_length = 1 * w
             cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
             dist_matrix = np.zeros((4, 1), dtype=np.float64)
             
             success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
             rmat, jac = cv2.Rodrigues(rot_vec)
             angles, _, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
             
             return {"pitch": angles[0]*360, "yaw": angles[1]*360, "roll": angles[2]*360}
         except Exception:
             return {"pitch":0, "yaw":0, "roll":0}

    # -------------------------------------------------------------------------
    # MODULE F: Noise Variance Analysis (New)
    # -------------------------------------------------------------------------
    def check_noise_consistency(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_std = np.std(laplacian)
            
            score = 0.0
            msg = "Consistent Noise Pattern"
            
            if noise_std < 5.0:
                score = 0.8 # Too smooth
                msg = "Suspiciously Smooth (Low Noise)"
            elif noise_std > 50.0:
                score = 0.3 # High ISO?
                msg = "High Noise Levels"
            else:
                score = 0.1
                
            return {"score": score, "noise_map": laplacian, "details": f"{msg} (Std: {noise_std:.1f})"}
        except Exception:
            return {"score": 0.5, "noise_map": np.zeros_like(image), "details": "Noise Check Failed"}

    # -------------------------------------------------------------------------
    # MODULE G: Texture Analysis (Gradient Proxy)
    # -------------------------------------------------------------------------
    def check_texture_artifacts(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            mean_mag = np.mean(magnitude)
            
            score = 0.0
            msg = "Natural Texture"
            if mean_mag < 10.0:
                score = 0.7
                msg = "Lack of Texture Detail"
            else:
                score = 0.1
                
            return {"score": score, "map": magnitude, "details": f"{msg} (Mag: {mean_mag:.1f})"}
        except Exception:
             return {"score": 0.5, "map": np.zeros_like(image), "details": "Texture Check Failed"}

    # -------------------------------------------------------------------------
    # MAIN: Weighted Probability Fusion (Universal Engine)
    # -------------------------------------------------------------------------
    def calculate_integrated_score(self, image, ai_prob):
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Multi-Face Detection
        results = self.face_mesh.process(rgb_image)
        landmarks_list = results.multi_face_landmarks
        
        # --- COMMON MODULES ---
        fft_res = self.check_frequency(image)
        ela_res = self.check_ela(image)
        noise_res = self.check_noise_consistency(image)
        tex_res = self.check_texture_artifacts(image)
        
        # --- BRANCHING LOGIC ---
        is_human_face = landmarks_list is not None and len(landmarks_list) > 0
        
        annotated_image = image.copy()
        veto_triggered = False
        veto_reason = ""
        face_count = 0
        
        if is_human_face:
            # === HUMAN MODE ===
            face_count = len(landmarks_list)
            max_bio_fake = 0.0
            worst_eye_details = ""
            
            for face_landmarks in landmarks_list:
                lm_list = face_landmarks.landmark
                eye = self.check_eye_biometrics(image, lm_list, w, h)
                current_bio_fake = 1.0 - eye["score"]
                
                if current_bio_fake > max_bio_fake:
                    max_bio_fake = current_bio_fake
                    worst_eye_details = eye["details"]
                
                # Draw Mesh
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

            bio_fake_prob = max_bio_fake
            
            # Weights: AI(0.5) + Bio(0.2) + Freq(0.1) + Pixel(0.1) + Noise(0.05) + Tex(0.05)
            global_score = (ai_prob * 0.5) + (bio_fake_prob * 0.2) + (fft_res["score"] * 0.1) + \
                           (ela_res["score"] * 0.1) + (noise_res["score"] * 0.05) + (tex_res["score"] * 0.05)
            
            if bio_fake_prob > 0.85:
                global_score = 0.99
                veto_triggered = True
                veto_reason = f"⚠️ BIOMETRIC VETO: {worst_eye_details}"
            
            mode_label = f"Human Forensic Mode ({face_count} detected)"
            bio_val = bio_fake_prob
            
        else:
            # === UNIVERSAL MODE ===
            # Weights: AI(0.4) + Freq(0.2) + Pixel(0.2) + Noise(0.1) + Tex(0.1) = 1.0
            bio_val = 0.0
            global_score = (ai_prob * 0.4) + (fft_res["score"] * 0.2) + (ela_res["score"] * 0.2) + \
                           (noise_res["score"] * 0.1) + (tex_res["score"] * 0.1)
            mode_label = "Universal Mode (Non-Human / No Face)"

        return {
            "global_score": global_score,
            "mode": mode_label,
            "face_count": face_count,
            "veto_triggered": veto_triggered,
            "veto_reason": veto_reason,
            "modules": {
                "Deep Learning (AI)": ai_prob,
                "Biological (Eye)": bio_val,
                "Frequency (FFT)": fft_res["score"],
                "Pixel Artifacts (ELA)": ela_res["score"],
                "Noise Variance": noise_res["score"],
                "Texture Analysis": tex_res["score"]
            },
            "data": {
                "fft": fft_res["spectrum"],
                "ela": ela_res["ela_image"],
                "mesh": annotated_image,
                "noise": noise_res["noise_map"],
                "texture": tex_res["map"],
                "pose": None 
            }
        }

    def calculate_fake_probability(self, image, ai_confidence=0.0):
        # Backward compatibility
        res = self.calculate_integrated_score(image, ai_confidence)
        return {
            "final_score": res["global_score"],
            "veto_triggered": res["veto_triggered"],
            "veto_reason": res["veto_reason"],
            "breakdown": res["modules"],
            "face_mesh_image": res["data"]["mesh"],
            "report_full": res
        }
