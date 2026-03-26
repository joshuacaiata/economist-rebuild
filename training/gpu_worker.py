# Stub file for gpu_worker.py
# This file is only used when remote_updates=True for distributed training
# For local training (remote_updates=False), this class is not actually used

class RemoteUpdater:
    """Stub class for distributed training. Not used when remote_updates=False."""
    
    def __init__(self, config, num_numeric):
        self.config = config
        self.num_numeric = num_numeric
        
    def update_shared_policy(self, data):
        """Stub method - should not be called when remote_updates=False"""
        raise NotImplementedError("RemoteUpdater should not be used when remote_updates=False")
