import time

class HardwareAwareNAS:
    """
    Neural Architecture Search (NAS) - Nightly loop.
    Profiles the inference latency at rest, and scales the model size (hidden dims, experts) 
    to saturate available capabilities (target_latency).
    """
    def __init__(self, model, target_latency=0.050, num_warmups=2, num_tests=5):
        self.model = model
        self.target_latency = target_latency
        self.num_warmups = num_warmups
        self.num_tests = num_tests

    def profile_latency(self):
        """
        Runs the model several times and averages execution time.
        """
        for _ in range(self.num_warmups):
            self.model.forward()
            
        total_time = 0.0
        for _ in range(self.num_tests):
            start = time.time()
            self.model.forward()
            total_time += (time.time() - start)
            
        return total_time / self.num_tests

    def nightly_loop(self):
        """
        Adjusts the model scale by measuring current latency and adapting.
        """
        current_latency = self.profile_latency()
        
        # 10% tolerance bounds
        lower_bound = self.target_latency * 0.90
        upper_bound = self.target_latency * 1.10
        
        # In a real model, this would invoke an expert addition/pruning protocol or dim scaling.
        # Here we mock it by calling a generic `resize` interface if implemented.
        if hasattr(self.model, 'resize'):
            if current_latency < lower_bound:
                # Upscale by the ratio
                ratio = self.target_latency / max(current_latency, 1e-5)
                new_size = int(self.model.size * min(ratio, 1.5)) # Cap scale at 1.5x max per night
                self.model.resize(new_size)
            elif current_latency > upper_bound:
                # Downscale
                ratio = self.target_latency / current_latency
                new_size = int(self.model.size * max(ratio, 0.5)) # Cap downscale
                self.model.resize(new_size)
