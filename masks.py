import numpy as np
import cv2




class Mask (object):
    def __init__ (self, Dx, Dy, frac):
        self.Dx = Dx
        self.Dy = Dy
        self.frac = frac
        self.area = self.Dx*self.Dy*self.frac
        self.mask = np.zeros((self.Dx, self.Dy), dtype=np.uint8)
    def __str__ (self):
        return f'Binary Mask: [{self.Dx},{self.Dy}]'



class RectangleMask (Mask):
    def __init__ (self, Dx, Dy, frac):
        """
        Create RectangularMask Object.
        ratio = base/height or height/base
        """
        super().__init__(Dx, Dy, frac)
        self.X0 = 0
        self.Y0 = 0
        self.base = 0
        self.ratio = 1.2
        self.height = 0

    def __call__(self):
        """
        Place A Circle Mask Randomly Within [self.Dx,self.Dy] Image.
        """
        assert (self.frac>0. and self.frac<1. and self.ratio>=1.)
        self.base = int(np.random.randint(self.area**0.5/self.ratio, np.minimum(self.Dx, self.area**0.5*self.ratio), 1))
        self.height = int(self.area/self.base)

        self.X0 = int(np.random.randint(0, self.Dx-self.base, 1))
        self.Y0 = int(np.random.randint(0, self.Dy-self.height, 1))
        
        self.mask = cv2.rectangle(self.mask, (self.X0, self.Y0), (self.X0+self.base, self.Y0+self.height), (1,0,0), thickness=-1)
        return np.array(1-self.mask, dtype=np.uint8)

    def __str__ (self):
        return f'Rectangular Binary Mask: [{self.Dx},{self.Dy}]'



class CircleMask (Mask):
    def __init__ (self, Dx, Dy, frac):
        """
        Create CircularMask Object.
        """
        super().__init__(Dx, Dy, frac)
        self.Xc = 0
        self.Yc = 0
        self.radius = int(np.sqrt(self.area/np.pi))

    def __call__(self):
        """
        Place A Circle Mask Randomly Within [self.Dx,self.Dy] Image.
        """
        assert (self.frac>0. and self.frac<1.)
        self.Xc = int(np.random.randint(self.radius, self.Dx-self.radius, 1))
        self.Yc = int(np.random.randint(self.radius, self.Dy-self.radius, 1))
        #print (xc, yc, self.radius)
        
        self.mask = cv2.circle(self.mask, (self.Xc, self.Yc), self.radius, (1,0,0), thickness=-1)
        return np.array(1-self.mask, dtype=np.uint8)

    def __str__ (self):
        return f'Circular Binary Mask: [{self.Dx},{self.Dy}]'



class IrregularMask (Mask):
    def __init__ (self, Dx, Dy, frac):
        """
        Create IrregularMask Object.
        """
        super().__init__(Dx, Dy, frac)

    def __call__(self):
        """
        Place A Binary Segment Randomly Within [self.Dx,self.Dy] Image.
        """
        assert (self.frac>0. and self.frac<1.)
        
        state_var = 0
        
        while (state_var<self.area):
            x_in = int(np.random.randint(0, self.Dx, 1))
            y_in = int(np.random.randint(0, self.Dy, 1))
            x_fin = x_in + int(np.random.randint(0, 100, 1))
            y_fin = y_in + int(np.random.randint(0, 100, 1))
            thickness = int(np.random.randint(2, 8, 1))
            self.mask = cv2.line (self.mask, (x_in,y_in), (x_fin, y_fin), (1,0,0), thickness)
            state_var = len(self.mask[self.mask==1])
        return np.array(1-self.mask, dtype=np.uint8)

    def __str__ (self):
        return f'Irregular Binary Mask: [{self.Dx},{self.Dy}]'








if __name__ == '__main__':
    circle = CircleMask(256, 256, 0.1)
    mask_circle = circle()
    print (circle)

    rectangle = RectangleMask(256, 256, 0.1)
    mask_rectangle = rectangle()
    print (rectangle)
    
    irregular = IrregularMask(256, 256, 0.1)
    mask_irregular = irregular()
    print (irregular)
