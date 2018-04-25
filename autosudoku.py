import pyautogui, time
from ocr import imageClass, OCRmodelClass, solveSudoku

if __name__ == '__main__':
	for i in range(2):
		loc = None
		while not loc:
			time.sleep(1)
			loc = pyautogui.locateOnScreen('button.png')
		# print(loc)


		x, y, w, h = loc[0]-540, loc[1]-430, 500, 500
		screenshot = pyautogui.screenshot(region=(x, y, w, h))

		im = imageClass()
		im.captureImage(screenshot)
		im.perspective()
		im.warp()

		ocr = OCRmodelClass()
		ocr.OCR_read(im)

		def callbk(image, xx, yy, num):
			pyautogui.moveTo(xx+x, yy+y)
			pyautogui.click()
			pyautogui.typewrite(num)
			time.sleep(0.01)

		solveSudoku(ocr.puzzle, im, callbk)
		# print(ocr.puzzle)
