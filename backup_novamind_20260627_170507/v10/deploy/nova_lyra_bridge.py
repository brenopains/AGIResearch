import numpy as np
import json
import os

class LyraBridge:
    """
    Orchestration layer connecting NovaMind's internal states to NVIDIA's Lyra 2.0
    Generative 3D World Model. Maps Action/Actor tensors into camera spatial poses
    and formats Symbolic primitives into scene generation prompts for Lyra's Cloud nodes.
    """
    def __init__(self, output_dir="cloud_payloads"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def prepare_trajectory(self, action_trace: list, run_id: str = "0"):
        """
        Converts continuous Action Tensors from the ActorCritic into a spatial 
        camera trajectory (.npz) formatted for Lyra-2/lyra2_custom_traj_inference.
        
        Args:
            action_trace: List of [10] dims tensors outputted by NovaMind.
        """
        num_frames = len(action_trace)
        
        # Define w2c (world-to-camera) N x 4 x 4 matrices
        w2c = np.zeros((num_frames, 4, 4), dtype=np.float32)
        
        # Base camera intrinsics N x 3 x 3
        intrinsics = np.zeros((num_frames, 3, 3), dtype=np.float32)
        
        x_cum, y_cum, z_cum = 0, 0, 0
        
        for i, action in enumerate(action_trace):
            # Map action tensor dims to spatial coordinates
            # Assuming action is normalized [-1, 1]
            x_cum += action[0] * 0.1 
            y_cum += action[1] * 0.1
            z_cum += action[2] * 0.1
            
            # Identity rotation, translating by accumulated actions
            mat = np.eye(4)
            mat[0, 3] = x_cum
            mat[1, 3] = y_cum
            mat[2, 3] = z_cum
            
            w2c[i] = mat
            
            # Standard intrinsic
            K = np.eye(3)
            K[0, 0] = 800 # focal length x
            K[1, 1] = 800 # focal length y
            K[0, 2] = 512 # cx
            K[1, 2] = 512 # cy
            intrinsics[i] = K
            
        payload_path = os.path.join(self.output_dir, f"lyra_trajectory_run_{run_id}.npz")
        np.savez(payload_path, w2c=w2c, intrinsics=intrinsics, image_height=1024, image_width=1024)
        return payload_path

    def prepare_captions(self, symbolic_programs: list, run_id: str = "0"):
        """
        Takes the abstract symbolic logic tree (AST primitives) emitted during
        the simulation and maps them to contextual text required for Lyra Chunk Generation.
        """
        captions = {}
        for i, prog_str in enumerate(symbolic_programs):
            # Frame index alignment logic (e.g. keyframes every 81 frames as per Lyra spec)
            frame_idx = str(i * 81)
            # Map structural program to generative text
            text_prompt = f"A generative scene governed by rules: {prog_str}"
            captions[frame_idx] = text_prompt
            
        payload_path = os.path.join(self.output_dir, f"lyra_captions_run_{run_id}.json")
        with open(payload_path, "w") as f:
            json.dump(captions, f, indent=4)
        return payload_path

# Export singleton
bridge = LyraBridge()
