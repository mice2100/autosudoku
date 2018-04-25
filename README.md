# auto sudoku resolver

[screen cast](screencast.webm)

*I created this project to learn OCR(with CNN). This includes everying:*
    * genFonts.py---- To create data for training
    * train.py ---- To train a CNN model and save to local folder for later use
    * autosudoku.py ---- Main controller, screen shot, call OCR to get puzzle, solve, then auto fill
    * ocr.py ---- Analysis the image and parse the puzzle
    * sudoku.py ---- Codes from http://norvig.com/sudoku.html
    * digitals/ ---- all train datas, output of genFonts.py
    * minst.h6 ---- Trained model, output of train.py
    * button.png ---- Part of sudoku game, autosudoku use it to determine if the game is activated. You NEED replace it with your own screen shot.
    
## This is tested on ubuntu 18.04.

## libraries used:
    * PIL
    * pyautogui
    * keras
    * numpy
    * opencv
  

*good luck*
