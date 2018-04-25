import glob, sys, os, fnmatch
from PIL import Image, ImageDraw, ImageFont

fnid = 0
folders = ['/usr/share/fonts/opentype/**', 
           '/usr/share/fonts/truetype/ubuntu/**']
for folder in folders:
    for fn in glob.glob(folder, recursive=True):
        if not fnmatch.fnmatch(fn, '*.tt?'):
            continue
        print(fn)

        # most ttc files include more that ONE fontface, so we try to use as much as 30
        for index in range(30):
            try:
                font = ImageFont.truetype(fn, 46, index)
                for txt in range(10):
                    pth = 'digitals/{}'.format(txt)
                    if not os.path.exists(pth):
                        os.mkdir(pth)
                    text = str(txt)
                    # the size font.getsize() returns not exactly same as the result, so we adjust it
                    tw, th = font.getsize(text)
                    im = Image.new('L',(tw+2, th+2))
                    draw = ImageDraw.Draw(im)
                    draw.text((1,1), text, 'white', font)
                    # get the box with pixels and fit in
                    bbox = im.getbbox()
                    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    size = max(w, h)
                    newim = Image.new('L', (size, size))
                    x, y = 0, 0
                    if w>h:
                        y = (w-h)//2
                    else:
                        x = (h-w)//2
                    newim.paste(im.crop(bbox), (x, y))
                    newim.resize((28,28)).save('{}/{}.jpg'.format(pth,fnid))
                fnid += 1
            except:
                print(sys.exc_info())
                break
