import cv2
import numpy as np
import keras
from PIL import Image

OUTPUT_SIZE = 600
class imageClass:
    def __init__(self):
        #.captured is the initially captured image
        self.captured = []
        #.gray is the grayscale captured image
        self.gray = []
        #.biggest contains a set of four coordinate points describing the
        self.biggest = None;
        #.output is an image resulting from the warp() method
        self.output = []
        #.mat is a matrix of 100 points found using a simple gridding algorithm
        #based on the four corner points from .biggest
        self.mat = np.zeros((100,2),np.float32)
        #.reshape is a reshaping of .mat
        self.reshape = np.zeros((100,2),np.float32)
        
    def captureImage(self, img):
        #captures the image and finds the biggest rectangle

        if isinstance(img, str):
            img = cv2.imread(img)
        else:
            img = np.array(img)
        self.captured = img

        self.gray = cv2.cvtColor(self.captured, cv2.COLOR_BGR2GRAY)

        #noise removal with gaussian blur
        # self.gray = cv2.GaussianBlur(self.gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(self.gray,255,1,1,11,2)

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #evaluate all blobs to find blob with biggest area
        #biggest rectangle in the image must be sudoku square
        self.biggest = None
        maxArea = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 30000: #50000 is an estimated value for the kind of blob we want to evaluate
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > maxArea and len(approx)==4:
                    self.biggest = approx
                    maxArea = area
        if maxArea > 0:
            #draw self.biggest approx contour
            cv2.polylines(self.captured,[self.biggest],True,(0,0,255),3)
            self.reorder() #reorder self.biggest
            print("Sudoku puzzle detected!")
            # cv2.imshow('sudoku', self.captured)
            # key = cv2.waitKey(10)
            # if key==27:
                # sys.exit()
            
    def reorder(self):
        #reorders the points obtained from finding the biggest rectangle
        #[top-left, top-right, bottom-right, bottom-left]
        a = self.biggest.reshape((4,2))
        b = np.zeros((4,2),dtype = np.float32)
     
        add = a.sum(1)
        b[0] = a[np.argmin(add)] #smallest sum
        b[2] = a[np.argmax(add)] #largest sum
             
        diff = np.diff(a,axis = 1) #y-x
        b[1] = a[np.argmin(diff)] #min diff
        b[3] = a[np.argmax(diff)] #max diff
        self.biggest = b

    def perspective(self):
        #create 100 points using "biggest" and simple gridding algorithm,
        #these 100 points define the grid of the sudoku puzzle
        #topLeft-topRight-bottomRight-bottomLeft = "biggest"
        # b = np.zeros((100,2),dtype = np.float32)
        c_sqrt=10
        if self.biggest.all() == None:
            self.biggest = [[0,0],[640,0],[640,480],[0,480]]
        tl,tr,br,bl = self.biggest[0],self.biggest[1],self.biggest[2],self.biggest[3]
        for k in range (0,100):
            i = k%c_sqrt
            j = (int)(k/c_sqrt)
            ml = [tl[0]+(bl[0]-tl[0])/9*j,tl[1]+(bl[1]-tl[1])/9*j]
            mr = [tr[0]+(br[0]-tr[0])/9*j,tr[1]+(br[1]-tr[1])/9*j]
            self.mat.itemset((k,0),ml[0]+(mr[0]-ml[0])/9*i)
            self.mat.itemset((k,1),ml[1]+(mr[1]-ml[1])/9*i)
        self.reshape = self.mat.reshape((c_sqrt,c_sqrt,2))

    def warp(self):
        #take distorted image and warp to flat square for clear OCR reading
        # mask = np.zeros((self.gray.shape),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        close = cv2.morphologyEx(self.gray,cv2.MORPH_CLOSE,kernel)
        division = np.float32(self.gray)/(close)
        result = np.uint8(cv2.normalize(division,division,0,255,cv2.NORM_MINMAX))
        result = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)

        output = np.zeros((OUTPUT_SIZE,OUTPUT_SIZE,3),np.uint8)
        c_sqrt=10
        for i,j in enumerate(self.mat):
            ri = (int)(i/c_sqrt)
            ci = i%c_sqrt
            def fn(x):
                return int(x*OUTPUT_SIZE/(c_sqrt-1))
            def fn2(x):
                return x*OUTPUT_SIZE/(c_sqrt-1)
            if ci != c_sqrt-1 and ri != c_sqrt-1:
                source = self.reshape[ri:ri+2, ci:ci+2 , :].reshape((4,2))
                dest = np.array( [ [fn2(ci),fn2(ri)],[fn2(ci+1),
                            fn2(ri)],[fn2(ci),fn2(ri+1)],
                            [fn2(ci+1),fn2(ri+1)] ], np.float32)
                trans = cv2.getPerspectiveTransform(source,dest)
                warp = cv2.warpPerspective(result,trans,(OUTPUT_SIZE,OUTPUT_SIZE))
                output[fn(ri):fn(ri+1) , fn(ci):fn(ci+1)] = warp[fn(ri):fn(ri+1) ,
                        fn(ci):fn(ci+1)].copy()
        self.output = output

class OCRmodelClass:
    #this class defines the data used for OCR,
    #and the associated methods for performing OCR
    def __init__(self):
        self.model = keras.models.load_model('minst.h6')
        
    def OCR_read(self,image):
        gray = Image.fromarray(image.output).convert('L')
        npgray = cv2.adaptiveThreshold(np.array(gray),255,1,1,11,2)

        self.puzzle = np.zeros((9,9),np.uint8)

        SIZE = OUTPUT_SIZE/9
        OFFSET = SIZE/10

        for xx in range(9):
            for yy in range(9):
                x = (int)(xx*SIZE+OFFSET)
                y = (int)(yy*SIZE+OFFSET)
                w = (int)(SIZE-OFFSET*2)
                h = (int)(SIZE-OFFSET*2)

                roi = npgray[y:y+h,x:x+w]
                img = Image.fromarray(roi)
                bbox = img.getbbox()
                if not bbox:
                    continue
                x += (bbox[0]-1)
                y += (bbox[1]-1)
                w = bbox[2]-bbox[0]+2
                h = bbox[3]-bbox[1]+2
                cv2.rectangle(image.output,(x,y),(x+w,y+h),(0,0,255),2)
                roi = npgray[y:y+h,x:x+w]
                roismall = np.array(cv2.resize(roi,(28,28)))
                roismall = roismall.reshape(1,28, 28, 1)
                roismall = roismall.astype('float32')
                roismall /= 255

                result = self.model.predict_classes(roismall)
                string = str(int((result[0])))
                if self.puzzle[yy,xx]==0:
                    self.puzzle[yy,xx] = int(string)
                cv2.putText(image.output,string,(x,y+h),0,1.4,(255,0,0),3)
 
        # print(self.puzzle)

def solveCallbk(image, x, y, num):
    cv2.putText(image.captured,num,(x,y),0,1.4,(0,0,255),2)

def solveSudoku(puzzle, image, fnCallback=None):
        import sudoku
        strpuz = [str(it) for it in puzzle.reshape(81)]
        grid = ''.join(strpuz)
        result = sudoku.solve(grid)
        if result==False:
            print("Can not be solved!!!", grid)
            return

        result = list(result.values())
        for xx in range(9):
            for yy in range(9):
                if puzzle[yy,xx]>0:
                    continue
                x = int(image.reshape[yy+1, xx][0])
                y = int(image.reshape[yy+1, xx][1])

                if fnCallback:
                    fnCallback(image, x+10, y-20, result[yy*9+xx])

if __name__ == '__main__':
    im = imageClass()
    im.captureImage('screenshot.png')
    im.perspective()
    # print(im.reshape)
    im.warp()

    ocr = OCRmodelClass()
    ocr.OCR_read(im)

    solveSudoku(ocr.puzzle, im, solveCallbk)
    cv2.imshow('final', im.captured)
    cv2.imwrite('output.jpg', im.captured)
    cv2.waitKey(20000)
