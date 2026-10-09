import sys
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
import os
os.makedirs('D:/temp/broca_v2', exist_ok=True)
from core.broca_v2.train import Trainer
t = Trainer()
t.train()
