from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage.filters import gaussian_filter
import skimage
import string
import random
import os
import math
from . import grid_distortion

#import pyvips

#https://stackoverflow.com/a/47381058/1018830
def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf, cmin=0, cmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, cmin,cmax,rmin,rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin,rmax,cmin,cmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, xx>=cmin, xx<cmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

#https://stackoverflow.com/a/47269413/1018830
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def rot_point(x,y,xc,yc,theta):
    x-=xc
    y-=yc
    h = math.sqrt(x**2+y**2)
    theta_o = math.atan2(y,x)
    x_n = h*math.cos(-theta+theta_o)
    y_n = h*math.sin(-theta+theta_o)

    return x_n+xc,y_n+yc

def tensmeyer_brightness(img, foreground=0, background=0):
    if len(img.shape)>2 and img.shape[2]==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret,th = cv2.threshold(gray ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th = (th.astype(np.float32) / 255)
    if len(img.shape)>2:
        th = th[...,None]

    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_tensmeyer_brightness(img, sigma=20, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0,sigma)
    background = random_state.normal(0,sigma)
    #print('fore {}, back {}'.format(foreground,background))

    img = tensmeyer_brightness(img, foreground, background)

    return img

class SyntheticText:

    def __init__(self,font_dir,text_dir,text_len=20,pad=20,line_prob=0.1,line_thickness=3,line_var=20,rot=10, gaus_noise=0.1, blur_size=1, hole_prob=0.2,hole_size=100,neighbor_gap_mean=20,neighbor_gap_var=7,use_warp=0.5,warp_std=1.5, warp_intr=12):
        self.font_dir = font_dir
        with open(os.path.join(font_dir,'fonts.list')) as f:
            self.fonts = f.read().splitlines()
        self.fontProbs = np.ones(len(self.fonts))/len(self.fonts) #init at uniform
        self.text_dir = text_dir
        with open(os.path.join(text_dir,'texts.list')) as f:
            self.texts = f.read().splitlines()
        self.text_len=text_len
        self.line_prob = line_prob
        self.pad = pad
        self.line_thickness = line_thickness
        self.line_var = line_var
        self.rot=rot
        self.gaus=gaus_noise
        self.blur_size=blur_size
        self.hole_prob=hole_prob
        self.hole_size=hole_size
        self.neighbor_gap_mean=neighbor_gap_mean
        self.neighbor_gap_var=neighbor_gap_var
        self.use_warp=use_warp
        self.warp_std=warp_std
        self.warp_intr=warp_intr

    
    def getFonts(self):
        return self.fonts.copy()

    def changeFontProb(self,f_index,amount):
        assert(amount>=0)
        self.fontProbs[f_index] += amount/len(self.fonts)
        self.fontProbs /= self.fontProbs.sum()
        #self.fontProbs = np.exp(self.fontProbs)/sum(np.exp(self.fontProbs)) #soft max

    def getText(self):
        #l = np.random.randint(1,20)
        #s = ''
        #for i in range(l):
        #    s += random.choice(string.ascii_letters)
        filename = random.choice(self.texts)
        with open(os.path.join(self.text_dir,filename)) as f:
            text = f.read().replace('\n',' ').replace('  ',' ')
        l = np.random.randint(1,self.text_len)
        start = np.random.randint(0,len(text)-l)
        t = text[start:start+l]
        return t

    def getFont(self,index=None):
        while True:
            if index is None:
                index = np.random.choice(len(self.fonts),p=self.fontProbs)
            filename = self.fonts[index] #random.choice(self.fonts)
            try:
                font = ImageFont.truetype(os.path.join(self.font_dir,filename), 100) 
                break
            except OSError:
                print('bad font: {}'.format(filename))
                index=None
        return font, index

    def getRenderedText(self,font=None,ink=None):
        f_index=-1
        for i in range(100):
            random_text = self.getText()
            if font is None:
                font, f_index = self.getFont()

            #create big canvas as it's hard to predict how large font will render
            size=(250+190*len(random_text),920)
            image = Image.new(mode='L', size=size)

            draw = ImageDraw.Draw(image)
            if ink is None:
                ink=(np.random.random()/2)+0.5
            try:
                draw.text((400, 250), random_text, font=font,fill=1)
            except OSError:
                font=None
                ink=None
                continue
            np_image = np.array(image)

            horzP = np.max(np_image,axis=0)
            minX=first_nonzero(horzP,0)
            maxX=last_nonzero(horzP,0)
            vertP = np.max(np_image,axis=1)
            minY=first_nonzero(vertP,0)
            maxY=last_nonzero(vertP,0)


            if (minX<maxX and minY<maxY):
                return np_image,random_text,minX,maxX,minY,maxY,font, f_index,ink

            if i>50:
                font=None
                ink=None
        return np.zeros((20,20))
        

    def getSample(self):
        while True:
            np_image,random_text,minX,maxX,minY,maxY,font, f_index,ink = self.getRenderedText()
            if np.random.random()<1.1: #above
                if  np.random.random()<0.5:
                    fontA=font
                else:
                    fontA=None
                np_imageA,random_textA,minXA,maxXA,minYA,maxYA,_,_,_ = self.getRenderedText(fontA,ink)
                gap = np.random.normal(self.neighbor_gap_mean,self.neighbor_gap_var)
                moveA = int(minY-gap)-maxYA
                mainY1=max(0,minYA+moveA)
                mainY2=maxYA+moveA
                AY1=maxYA-(mainY2-mainY1)
                AY2=maxYA
                AxOff = np.random.normal(-30,self.neighbor_gap_var*5)
                mainCenter = (maxX+minX)//2
                ACenter = (maxXA+minXA)//2
                AxOff = int((mainCenter-ACenter)+AxOff)
                mainX1 = max(0,minXA+AxOff)
                mainX2 = min(np_image.shape[1]-1,maxXA+AxOff)
                if mainX2>mainX1:
                    AX1 = minXA-(minXA+AxOff-mainX1)
                    AX2 = maxXA-(maxXA+AxOff-mainX2)
                    #print('[{}:{},{}:{}] [{}:{},{}:{}]'.format(mainY1,mainY2+1,mainX1,mainX2+1,AY1,AY2+1,AX1,AX2+1))
                    np_image[mainY1:mainY2+1,mainX1:mainX2+1] = np.maximum(np_image[mainY1:mainY2+1,mainX1:mainX2+1],np_imageA[AY1:AY2+1,AX1:AX2+1])
            if np.random.random()<1.1: #below
                if  np.random.random()<0.5:
                    fontA=font
                else:
                    fontA=None
                np_imageA,random_textA,minXA,maxXA,minYA,maxYA,_,_,_ = self.getRenderedText(fontA,ink)
                gap = np.random.normal(self.neighbor_gap_mean,self.neighbor_gap_var)
                moveA = int(maxY+gap)-minYA
                mainY1=minYA+moveA
                mainY2=min(np_image.shape[0]-1,maxYA+moveA)
                AY1=minYA
                AY2=minYA+(mainY2-mainY1)
                AxOff = np.random.normal(30,self.neighbor_gap_var*5)
                mainCenter = (maxX+minX)//2
                ACenter = (maxXA+minXA)//2
                AxOff = int((mainCenter-ACenter)+AxOff)
                mainX1 = max(0,minXA+AxOff)
                mainX2 = min(np_image.shape[1]-1,maxXA+AxOff)
                if mainX2>mainX1:
                    AX1 = minXA-(minXA+AxOff-mainX1)
                    AX2 = maxXA-(maxXA+AxOff-mainX2)
                    #print('[{}:{},{}:{}] [{}:{},{}:{}]'.format(mainY1,mainY2+1,mainX1,mainX2+1,AY1,AY2+1,AX1,AX2+1))
                    np_image[mainY1:mainY2+1,mainX1:mainX2+1] = np.maximum(np_image[mainY1:mainY2+1,mainX1:mainX2+1],np_imageA[AY1:AY2+1,AX1:AX2+1])


            #base_image = pyvips.Image.text(random_text, dpi=300, font=random_font)
            #org_h = base_image.height
            #org_w = base_image.width
            #np_image = np.ndarray(buffer=base_image.write_to_memory(),
            #        dtype=format_to_dtype[base_image.format],
            #        shape=[base_image.height, base_image.width, base_image.bands])


            np_image=np_image*0.8
            padding=np.random.normal(0,self.pad,(2,2))
            padding[padding<0]*=0.75
            padding = np.round(padding)#.astype(np.uint8)

            #lines
            while np.random.rand() < self.line_prob:
                side = np.random.choice([1,2,3,4])
                if side==1: #bot
                    y1 = np.random.normal(maxY+20,self.line_var)
                    y2 = y1+np.random.normal(0,self.line_var/4)
                    x1 = np.random.normal(minX-padding[1,0],self.line_var*2)
                    x2 = np.random.normal(maxX+padding[1,1],self.line_var*2)
                elif side==2: #top
                    y1 = np.random.normal(minY-20,self.line_var)
                    y2 = y1+np.random.normal(0,self.line_var/4)
                    x1 = np.random.normal(minX-padding[1,0],self.line_var*2)
                    x2 = np.random.normal(maxX+padding[1,1],self.line_var*2)
                elif side==3: #left
                    x1 = np.random.normal(minX-20,self.line_var)
                    x2 = x1+np.random.normal(0,self.line_var/4)
                    y1 = np.random.normal(minY-padding[0,0],self.line_var*2)
                    y2 = np.random.normal(maxY+padding[0,1],self.line_var*2)
                elif side==4: #right
                    x1 = np.random.normal(maxX+20,self.line_var)
                    x2 = x1+np.random.normal(0,self.line_var/4)
                    y1 = np.random.normal(minY-padding[0,0],self.line_var*2)
                    y2 = np.random.normal(maxY+padding[0,1],self.line_var*2)
                thickness = np.random.random()*(self.line_thickness-1) + 1
                yy,xx,val = weighted_line(y1,x1,y2,x2,thickness,0,np_image.shape[0],0,np_image.shape[1])
                color = np.random.random()
                np_image[yy,xx]=np.maximum(val*color,np_image[yy,xx])
                #print('line {}:  {},{}  {},{}'.format(side,x1,y1,x2,y2))

            #crop region
            #np_image = np.pad(np_image,padding,mode='constant')*0.8
            minY = max(0,minY-padding[0,0])
            minX = max(0,minX-padding[1,0])
            maxY = maxY+1+padding[0,1]
            maxX = maxX+1+padding[1,1]

            #rot
            #xc=(maxX+minX)//2
            #yc=(maxY+minY)//2
            ##actually center the image on xc,yc
            #removeLeft = xc-min(xc,np_image.shape[1]-xc)
            #removeRight = (np_image.shape[1]-xc)-min(xc,np_image.shape[1]-xc)
            #removeTop = yc-min(yc,np_image.shape[0]-yc)
            #removeBot = (np_image.shape[0]-yc)-min(yc,np_image.shape[0]-yc)
            #np_image = np_image[removeTop:-removeBot,removeLeft:-removeRight]
            #minY -= removeTop
            #maxY -= removeTop+removeBot
            #minX -= removeLeft
            #maxX -= removeLeft+removeRight
            degrees=np.random.normal(0,self.rot)
            degrees = max(-2.5*self.rot,degrees)
            degrees = min(2.5*self.rot,degrees)
            np_image = rotate(np_image,degrees,reshape=False)

            
            theta = math.pi*degrees/180
            xc=np_image.shape[1]/2
            yc=np_image.shape[0]/2
            tlX,tlY = rot_point(minX,minY,xc,yc,theta)
            trX,trY = rot_point(maxX,minY,xc,yc,theta)
            blX,blY = rot_point(minX,maxY,xc,yc,theta)
            brX,brY = rot_point(maxX,maxY,xc,yc,theta)
            #w=(maxX-minX)/2
            #h=(maxY-minY)/2
            #xc=(maxX+minX)/2
            #yc=(maxY+minY)/2
            #tlX =-w*math.cos(theta) - h*math.sin(theta) + xc
            #trX = w*math.cos(theta) - h*math.sin(theta) + xc
            #brX = w*math.cos(theta) + h*math.sin(theta) + xc
            #blX =-w*math.cos(theta) + h*math.sin(theta) + xc
            #tlY =-w*math.sin(theta) + h*math.cos(theta) + yc
            #trY = w*math.sin(theta) + h*math.cos(theta) + yc
            #brY = w*math.sin(theta) - h*math.cos(theta) + yc
            #blY =-w*math.sin(theta) - h*math.cos(theta) + yc

            minY = int(round(max(min(tlY,trY,blY,brY),0)))
            maxY = int(round(min(max(tlY,trY,blY,brY),np_image.shape[0])))
            minX = int(round(max(min(tlX,trX,blX,brX),0)))
            maxX = int(round(min(max(tlX,trX,blX,brX),np_image.shape[1])))
            

            #yy,xx,val = weighted_line(minY,minX,minY,maxX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val
            #yy,xx,val = weighted_line(minY,maxX,maxY,maxX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val
            #yy,xx,val = weighted_line(maxY,maxX,maxY,minX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val
            #yy,xx,val = weighted_line(maxY,minX,minY,minX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val

            #yy,xx,val = weighted_line(tlY,tlX,trY,trX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val
            #yy,xx,val = weighted_line(trY,trX,brY,brX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val
            #yy,xx,val = weighted_line(brY,brX,blY,blX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val
            #yy,xx,val = weighted_line(blY,blX,tlY,tlX,20,0,np_image.shape[0],0,np_image.shape[1])
            #np_image[yy,xx]=val


            np_image = np_image[minY:maxY,minX:maxX]

            if np_image.shape[1]==0 or np_image.shape[0]==0:
                continue

            #holes
            while np.random.rand() < self.hole_prob:
                x=np.random.randint(0,np_image.shape[1])
                y=np.random.randint(0,np_image.shape[0])
                rad = np.random.randint(1,self.hole_size)
                rad2 = np.random.randint(rad/3,rad)
                size = rad*rad2
                rot = np.random.random()*2*np.pi
                strength = (1.6*np.random.random()-1.0)*(1-size/(self.hole_size*self.hole_size))
                yy,xx = skimage.draw.ellipse(y, x, rad, rad2, shape=np_image.shape, rotation=rot)
                complete = np.random.random()
                app = np.maximum(1-np.abs(np.random.normal(0,1-complete,yy.shape)),0)
                np_image[yy,xx] = np.maximum(np.minimum(np_image[yy,xx]+strength*app,1),0)


            #noise
            #specle noise
            #gaus_n = 0.2+(self.gaus-0.2)*np.random.random()
            gaus_n = abs(np.random.normal(self.gaus,0.1))
            if gaus_n==0:
                gaus_n=0.00001
            
            np_image += np.random.normal(0,gaus_n,np_image.shape)
            #blur
            blur_s = np.random.normal(self.blur_size,0.2)
            np_image = gaussian_filter(np_image,blur_s)

            minV = np_image.min()
            maxV = np_image.max()
            np_image = (np_image-minV)/(maxV-minV)

            #contrast/brighness
            cv_image = (255*np_image).astype(np.uint8)
            cv_image = apply_tensmeyer_brightness(cv_image,25)
            #warp aug
            if random.random() < self.use_warp:
                if type(self.warp_intr) is list:
                    intr = np.random.randint(self.warp_intr[0],self.warp_intr[1])
                else:
                    intr=self.warp_intr
                if type(self.warp_std) is list:
                    std = (self.warp_std[1]-self.warp_std[0])*np.random.random()+self.warp_std[0]
                else:
                    std=self.warp_std
                std *= intr/12
                cv_image = grid_distortion.warp_image(cv_image,w_mesh_std=std,h_mesh_std=std,w_mesh_interval=intr,h_mesh_interval=intr)
            np_image =cv_image/255.0

            if np_image.shape[0]>0 and np_image.shape[1]>1:
                break

            #random pad
            #if random.random()>0.5:
            #    w = random.randint(0,np_img.shape[0])
            #    if random.random()>0.5:
            #        pad = 

        return np_image, random_text, f_index

    def getFixedSample(self,text,fontfile):
        #create big canvas as it's hard to predict how large font will render
        font = self.getFont(fontfile)
        size=(250+190*len(text),920)
        image = Image.new(mode='L', size=size)

        draw = ImageDraw.Draw(image)
        ink=(0.5/2)+0.5
        draw.text((400, 250), text, font=font,fill=1)
        np_image = np.array(image)

        horzP = np.max(np_image,axis=0)
        minX=first_nonzero(horzP,0)
        maxX=last_nonzero(horzP,0)
        vertP = np.max(np_image,axis=1)
        minY=first_nonzero(vertP,0)
        maxY=last_nonzero(vertP,0)


        np_image=np_image*0.8
        padding=np.array([[5,5],[5,5]])
        #padding[padding<0]*=0.75
        #padding = np.round(padding)#.astype(np.uint8)


        #crop region
        #np_image = np.pad(np_image,padding,mode='constant')*0.8
        minY = max(0,minY-padding[0,0])
        minX = max(0,minX-padding[1,0])
        maxY = maxY+1+padding[0,1]
        maxX = maxX+1+padding[1,1]

        #rot


        np_image = np_image[minY:maxY,minX:maxX]

        #noise
        #specle noise
        #gaus_n = 0.2+(self.gaus-0.2)*np.random.random()
        gaus_n=0.001
        
        np_image += np.random.normal(0,gaus_n,np_image.shape)
        #blur
        blur_s = np.random.normal(self.blur_size,0.2)
        np_image = gaussian_filter(np_image,blur_s)

        minV = np_image.min()
        maxV = np_image.max()
        np_image = (np_image-minV)/(maxV-minV)

        #contrast/brighness
        #cv_image = (255*np_image).astype(np.uint8)
        #cv_image = apply_tensmeyer_brightness(cv_image,25)
        #np_image =cv_image/255.0



        return np_image
