import json

import fire
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from google.protobuf import text_format

import simpy

from second.dynamic.devices import SynthiaGTState, LightCurtain, SETVelocity
from second.dynamic.kittiviewer.backend import vizstream

app = Flask("second")
CORS(app)

class SecondBackend:
    def __init__(self):
        self.env = simpy.Environment()
        self.stop_event = self.env.event()

        # devices
        self.gt_state_device = SynthiaGTState(self.env, "train", cam=True, colored_pc=True)
        self.light_curtain = LightCurtain(self.env, self.gt_state_device)
        self.set_reactive = SETVelocity(self.env, self.light_curtain)

        self.devices = [self.gt_state_device, self.light_curtain, self.set_reactive]
        print("BACKEND: all devices created.")

        # Register device streams for visualization.
        vizstream(app, self.gt_state_device.stream, astype="scene_cloud")
        # vizstream(app, self.gt_state_device.stream, astype="camera_image")
        vizstream(app, self.light_curtain.stream,   astype="lc_cloud")

    def clear_simulation(self):
        del self.env, self.stop_event
        self.env = simpy.Environment()
        self.stop_event = self.env.event()
        for device in self.devices:
            device.reset(self.env)
    
    def run_simulation(self, idx):
        if self.env.peek() != float("inf"):
            raise Exception("New simulation requested without stopping previous simulation!" + 
                            " Use <Stop Simulation> button.")

        # create process
        gt_state_process = self.env.process(self.gt_state_device.process(idx, preload=True))
        set_reactive_process = self.env.process(self.set_reactive.process())
        
        # run simulation
        self.env.run(until=gt_state_process | self.stop_event)

        # clear simulation
        self.clear_simulation()
    
    def stop_simulation(self):
        self.stop_event.succeed()


BACKEND = SecondBackend()

@app.route('/api/run_simulation', methods=['GET', 'POST'])
def run_simulation():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    video_idx = instance["video_idx"]
    enable_int16 = instance["enable_int16"]
    
    BACKEND.run_simulation(video_idx)
    
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    
    BACKEND.stop_simulation()
    
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

def main(port=16666):
    app.run(host='127.0.0.1', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
