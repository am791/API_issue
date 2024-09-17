from ultralytics import YOLO

#seg_model = YOLO('yolov8n-seg.pt')
#seg_model.train(data='/home/enflame/OCR_IND/API_issue/datasets/segmentation/data.yaml', task='detect', epochs=50, imgsz=640, batch=8, name='yolo_seg', project= '/home/enflame/OCR_IND/train_models')


#det_model = YOLO('yolov8n.pt')
#det_model.train(data='/home/enflame/OCR_IND/API_issue/datasets/detection/data.yaml', epochs=500, imgsz=640, batch=8, patience=30, name='yolo_det', project= '/home/enflame/OCR_IND/train_models')

seg_model = YOLO('/home/enflame/OCR_IND/train_models/seg/best.pt')
results_seg = seg_model.predict(source='/home/enflame/OCR_IND/API_issue/test_invoice.png', save=True)

det_model = YOLO('/home/enflame/OCR_IND/train_models/det/best.pt')
results_det = det_model.predict(source='/home/enflame/OCR_IND/API_issue/test_invoice.png', imgsz=640, save=True)
