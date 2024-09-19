import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import json
import subprocess
import os

class_mapping = {
    0: 'BuyerAddress',
    1: 'BuyerName',
    2: 'BuyerTaxID',
    3: 'Date',
    4: 'InvoiceNumber',
    5: 'ProductGrossWorth',
    6: 'ProductName',
    7: 'ProductNetPrice',
    8: 'ProductNetWorth',
    9: 'ProductQuantity',
    10: 'ProductUM',
    11: 'ProductVAT',
    12: 'Seller IBAN',
    13: 'SellerAddress',
    14: 'SellerName',
    15: 'SellerTaxID',
    16: 'TotalGrossWorth',
    17: 'TotalNetWorth',
    18: 'TotalVAT',
    19: 'TotalVATPercent'
}


def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

seg_model = YOLO('/home/enflame/OCR_IND/train_models/seg/best.pt')
det_model = YOLO('/home/enflame/OCR_IND/train_models/det/best.pt')


#seg_model = YOLO('/content/drive/MyDrive/OWN_OCR/OCR/outputs/yolo_seg3/weights/best.pt')
#det_model = YOLO('/content/drive/MyDrive/best.pt')  # 100 epoc offline trained on 230 images

#img = cv2.imread('/content/drive/MyDrive/OWN_OCR/OCR/1.png')

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def inference(imag):
    img = cv2.imread(imag)
    data_extracted = {
        'BuyerAddress': '',
        'BuyerName': '',
        'BuyerTaxID': '',
        'Date': '',
        'InvoiceNumber': '',
        'ProductGrossWorth': [],
        'ProductName': [],
        'ProductNetPrice': [],
        'ProductNetWorth': [],
        'ProductQuantity': [],
        'ProductUM': [],
        'ProductVAT': [],
        'Seller IBAN': '',
        'SellerAddress': '',
        'SellerName': '',
        'SellerTaxID': '',
        'TotalGrossWorth': '',
        'TotalNetWorth': '',
        'TotalVAT': '',
        'TotalVATPercent': ''
    }

    json_data = {
        'BuyerAddress': data_extracted['BuyerAddress'],
        'BuyerName': data_extracted['BuyerName'],
        'BuyerTaxID': data_extracted['BuyerTaxID'],
        'Date': data_extracted['Date'],
        'InvoiceNumber': data_extracted['InvoiceNumber'],
        'items': [],
        'Seller IBAN': data_extracted['Seller IBAN'],
        'SellerAddress': data_extracted['SellerAddress'],
        'SellerName': data_extracted['SellerName'],
        'SellerTaxID': data_extracted['SellerTaxID'],
        'TotalGrossWorth': data_extracted['TotalGrossWorth'],
        'TotalNetWorth': data_extracted['TotalNetWorth'],
        'TotalVAT': data_extracted['TotalVAT'],
        'TotalVATPercent': data_extracted['TotalVATPercent']
    }

    results_seg = seg_model(img)

    for i, box in enumerate(results_seg[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        roi = img[y1:y2, x1:x2]

        results_det = det_model(roi)
        detections = []

        for result in results_det:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                class_name = class_mapping.get(class_id)

                if class_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_img = img[y1:y2, x1:x2]
                    detections.append({
                        'class_name': class_name,
                        'bbox': [x1, y1, x2, y2],
                        'cropped_img': cropped_img
                    })

        detections = sorted(detections, key=lambda det: det['bbox'][1])

        for det in detections:
            class_name = det['class_name']
            cropped_img = det['cropped_img']

            print(f"\nOCR Results for {class_name}:")
            result = ocr.ocr(cropped_img, cls=True)
            res = result[0]
            text = ""
            if res is not None:
                for item in res:
                    text = text + " " + item[1][0]
                if class_name.startswith("Product"):
                    data_extracted[class_name].append(text.strip())
                else:
                    data_extracted[class_name] = text.strip()

    json_data.update({
        'BuyerAddress': data_extracted['BuyerAddress'],
        'BuyerName': data_extracted['BuyerName'],
        'BuyerTaxID': data_extracted['BuyerTaxID'],
        'Date': data_extracted['Date'],
        'InvoiceNumber': data_extracted['InvoiceNumber'],
        'Seller IBAN': data_extracted['Seller IBAN'],
        'SellerAddress': data_extracted['SellerAddress'],
        'SellerName': data_extracted['SellerName'],
        'SellerTaxID': data_extracted['SellerTaxID'],
        'TotalGrossWorth': data_extracted['TotalGrossWorth'],
        'TotalNetWorth': data_extracted['TotalNetWorth'],
        'TotalVAT': data_extracted['TotalVAT'],
        'TotalVATPercent': data_extracted['TotalVATPercent']
    })

    num_products = len(data_extracted['ProductName'])

    # Iterate through the product-related fields and group them into a dictionary for each product
    for i in range(num_products):
        product = {
            'ProductName': data_extracted['ProductName'][i],
            'ProductQuantity': data_extracted['ProductQuantity'][i],
            'ProductNetPrice': data_extracted['ProductNetPrice'][i],
            'ProductGrossWorth': data_extracted['ProductGrossWorth'][i],
            'ProductNetWorth': data_extracted['ProductNetWorth'][i],
            'ProductUM': data_extracted['ProductUM'][i],
            'ProductVAT': data_extracted['ProductVAT'][i]
        }
        json_data['items'].append(product)

    print("\nTHE FINAL DATA EXTRACTED: ", json.dumps(json_data, indent=4, ensure_ascii=False))

cwd = os.getcwd()
subprocess.run(f'find /home/enflame/OCR_IND/API_issue/test -name "*" 2>&1 | tee {cwd}/test_images.txt', shell=True)

with open(f'{cwd}/test_images.txt', 'r') as f:
    imgs = f.readlines()[1:]

for img in imgs:
    print("SERVICE STARTED FOR: ", img)
    result = inference(img.strip())
