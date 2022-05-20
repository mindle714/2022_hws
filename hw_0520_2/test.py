from struct import unpack

'''
marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        

if __name__ == "__main__":
    img = JPEG('lena.jpeg')
    img.decode()    

# OUTPUT:
# Start of Image
# Application Default Header
# Quantization Table
# Quantization Table
# Start of Frame
# Huffman Table
# Huffman Table
# Huffman Table
# Huffman Table
# Start of Scan
# End of Image
'''

import numpy as np
from scipy.fftpack import fft, dct, idct
import cv2
np.set_printoptions(precision=2, suppress=True)

dct_opt=dict(norm='ortho')

def _dct(e):
#  return cv2.dct(e)
  return dct(dct(e, axis=0, **dct_opt), axis=1, **dct_opt)

def _idct(e):
#  return cv2.idct(e)
  return idct(idct(e, axis=0, **dct_opt), axis=1, **dct_opt)

_shift_test = np.array([
  [8, 14, 23, 37, 52, 68, 73, 82],
  [6, 14, 24, 37, 46, 67, 74, 81],
  [3, 11, 28, 35, 48, 62, 72, 82],
  [4, 13, 22, 28, 44, 61, 69, 86],
  [5, 11, 18, 30, 40, 59, 72, 86],
  [5, 9, 16, 29, 39, 58, 74, 83],
  [-1, 8, 16, 31, 38, 59, 75, 80],
  [2, 11, 18, 30, 37, 57, 69, 82]
])

_test = _shift_test + 128.

print(_dct(_shift_test))

_quant = np.array([
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
])

_q = np.round(_dct(_shift_test) / _quant);
res = np.round(_idct(_q * _quant)) + 128;
print(res)

print(np.sqrt(np.mean((_test - res)**2)))
