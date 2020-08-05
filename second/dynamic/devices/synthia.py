import tqdm

from second.dynamic.data.synthia import SynthiaVideoDataset
from second.dynamic.devices.device import Device

class SynthiaGTState(Device):
    def __init__(self, env, split, latency=40, cam=False, colored_pc=False):
        """
        Args:
            env: (simply.Environment) simulation environment.
            split: (string) dataset split ["train"/"test"].
            latency: (float) latency of the Synthia dataset (25fps).
            cam: (bool) whether to publish camera image
            colored_pc: (bool) whether to colorize points
        """
        super(SynthiaGTState, self).__init__(env, capacity=1)  # synthia will only expose the most recent ground truth
        self.dataset = SynthiaVideoDataset(split, cam, colored_pc)
        self.latency = latency
    
    def process(self, idx, preload=False):
        self.stream.clear()  # empty stream

        video = self.dataset[idx]
        if preload:
            print("Preloading video ...")
            video = [video[i] for i in tqdm.tqdm(range(len(video)))]

        for i in range(len(video)):
            frame_gt = video[i]
            self.publish(frame_gt)  # first publication is at t=0
            yield self.env.timeout(self.latency)
