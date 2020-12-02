from ctypes import *

so_file = './cDescent.so'
cfactorial = CDLL(so_file)