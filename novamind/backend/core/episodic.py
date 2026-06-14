import time

class EpisodicMemory:
    """
    Layer 2: Episodic Memory.
    Stores the timeline of events. Raw events can decay/expire over time,
    but they should be compressed into World Models before they do.
    """
    def __init__(self):
        self.timeline = []
        self.max_memory_size = 5000  # "O bruto pode expirar"

    def record_event(self, event):
        """
        Takes an event from Sensory Flow and adds it to the episodic timeline.
        """
        episode = {
            "episode_id": f"ep_{len(self.timeline)}",
            "event": event,
            "recorded_at": time.time(),
            "compressed": False
        }
        self.timeline.append(episode)
        
        # Simulating decay/forgetting of raw frames
        if len(self.timeline) > self.max_memory_size:
            # We remove the oldest uncompressed ones, or just the oldest ones.
            # In a real system, we'd ensure they've been compressed to the World Model.
            self.timeline.pop(0)
            
        return episode

    def get_uncompressed_episodes(self):
        return [ep for ep in self.timeline if not ep["compressed"]]

    def mark_compressed(self, episode_id):
        for ep in reversed(self.timeline):
            if ep["episode_id"] == episode_id:
                ep["compressed"] = True
                break

    def get_timeline(self):
        return self.timeline
