import time

class SensoryFlow:
    """
    Layer 1: Receives raw streams (Text, Audio, Vision) and standardizes them into Observation Events.
    """
    def __init__(self):
        self.stream = []

    def process_input(self, raw_data):
        """
        raw_data: dict containing 'type' (text/audio/vision) and 'content'
        Returns a standardized event dict.
        """
        event = {
            "id": f"evt_{int(time.time() * 1000)}",
            "type": raw_data.get("type", "unknown"),
            "content": raw_data.get("content", ""),
            "timestamp": time.time(),
            "source": raw_data.get("source", "unknown")
        }
        self.stream.append(event)
        # Keep only the latest 1000 items in sensory buffer
        if len(self.stream) > 1000:
            self.stream.pop(0)
            
        return event

    def get_recent(self, count=10):
        return self.stream[-count:]
