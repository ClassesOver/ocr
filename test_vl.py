from paddleocr import PaddleOCRVL



ocr = PaddleOCRVL()

results = ocr.predict(r'D:\ocr\ocrv5\ocr\code\images\stock_v1\line.png')

print(results)