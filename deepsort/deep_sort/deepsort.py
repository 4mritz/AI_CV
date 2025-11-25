import numpy as np
from deepsort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deepsort.deep_sort.tracker import Tracker
from deepsort.deep_sort.detection import Detection
from deepsort.tools import generate_detections as gdet

class DeepSort:
    def __init__(self, model_filename, max_cosine_distance=0.4, nn_budget=None):
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update_tracks(self, bbox_xywh, confidences, classes, frame):
        features = self.encoder(frame, bbox_xywh)
        detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(bbox_xywh, confidences, features)]

        self.tracker.predict()
        self.tracker.update(detections)

        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            bbox = track.to_tlbr()
            tracks.append([*bbox, track.track_id])

        return tracks
