import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import json

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

data_extracted = {
    'BuyerAddress': '',
    'BuyerName': '',
    'BuyerTaxID': '',
    'Date': '',
    'InvoiceNumber': '',
    'ProductGrossWorth': '',
    'ProductName': '',
    'ProductNetPrice': '',
    'ProductNetWorth': '',
    'ProductQuantity': '',
    'ProductUM': '',
    'ProductVAT': '',
    'Seller IBAN': '',
    'SellerAddress': '',
    'SellerName': '',
    'SellerTaxID': '',
    'TotalGrossWorth': '',
    'TotalNetWorth': '',
    'TotalVAT': '',
    'TotalVATPercent': ''

}

def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

seg_model = YOLO('/home/enflame/OCR_IND/train_models/seg/best.pt')
det_model = YOLO('/home/enflame/OCR_IND/train_models/det/best.pt')

img = cv2.imread('/home/enflame/OCR_IND/API_issue/test_invoice.png')

ocr = PaddleOCR(use_angle_cls=True, lang='en')
results_seg = seg_model(img)

for i, box in enumerate(results_seg[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    roi = img[y1:y2, x1:x2]

    results_det = det_model(roi)
    cropped_images_by_class = {key: [] for key in class_mapping.values()}

    for result in results_det:
      boxes = result.boxes
      for box in boxes:
          class_id = int(box.cls)
          class_name = class_mapping.get(class_id)

          if class_name:
              x1, y1, x2, y2 = map(int, box.xyxy[0])
              cropped_img = img[y1:y2, x1:x2]
              cropped_images_by_class[class_name].append(cropped_img)

    for class_name, images in cropped_images_by_class.items():
        print(f"\nOCR Results for {class_name}:")
        if(len(images)==0):
            print("  [ppocr]: Not detected")
        for cropped_img in images:
            # display_image(cropped_img, title=f"OCR ON {class_name}")
            result = ocr.ocr(cropped_img, cls=True)
            res = result[0]
            text = ""
            if res is not None:
                for item in res:
                    text = text + " " + item[1][0]
            data_extracted[class_name]=text.strip()


print("\nTHE FINAL DATA EXTRACTED: ", json.dumps(data_extracted, indent=4, ensure_ascii=False))
