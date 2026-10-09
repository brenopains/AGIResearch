import pytest
import time
from novamind.v10.models.nas import HardwareAwareNAS

class MockResizableModel:
    def __init__(self):
        self.size = 100
        
    def forward(self):
        # Mocks a computation that takes time proportional to size
        time.sleep(self.size * 0.0001)
        return True
        
    def resize(self, new_size):
        self.size = new_size

def test_hardware_nas():
    model = MockResizableModel()
    
    # Target latency of 0.015 seconds per forward pass
    nas = HardwareAwareNAS(model, target_latency=0.015)
    
    # Run the nightly loop evaluation
    initial_size = model.size
    
    nas.nightly_loop()
    
    final_size = model.size
    
    # Originally size=100 -> takes ~0.010s
    # Target is 0.015s, so the model should be UPSCALED (size > 100)
    assert final_size > initial_size, "NAS should upscale model to utilize target latency"
    
def test_hardware_nas_downscale():
    model = MockResizableModel()
    model.resize(300) # Takes ~0.030s
    
    nas = HardwareAwareNAS(model, target_latency=0.010)
    nas.nightly_loop()
    
    assert model.size < 300, "NAS should downscale model if latency exceeds target bounds"
