
print("Initializing Forensics Debug...")
import cv2
import numpy as np
try:
    from forensics import ForensicAnalyzer
    print("Forensics Imported.")
    
    analyzer = ForensicAnalyzer()
    print("Analyzer Initialized.")
    
    # Create dummy image (blue square)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(img, (100,100), (300,300), (255,0,0), -1)
    
    print("Running integrated score...")
    report = analyzer.calculate_integrated_score(img, 0.5)
    print("Success!")
    print("Keys:", report.keys())
    print("Data Keys:", report["data"].keys())
    
except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
