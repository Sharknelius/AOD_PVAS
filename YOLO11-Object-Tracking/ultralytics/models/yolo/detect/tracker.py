import cv2
import pandas as pd
from datetime import datetime
import numpy as np
from time import time
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class ObjectCounter(BaseSolution):
    def __init__(self, output_filename="output/output.avi", **kwargs):
        super().__init__(**kwargs)
        self.counted_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.spd = {}
        self.trk_pt = {}
        self.trk_pp = {}
        self.trk_pa = {}
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)

        # Initialize video writer
        self.video_writer = None
        self.output_filename = output_filename

    def initialize_writer(self, width, height):
        """Initialize video writer if not already initialized."""
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # Use XVID or MJPG codec
            self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, 20.0, (width, height))

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """Count objects and update file based on centroid movements."""
        if prev_position is None or track_id in self.counted_ids:
            return

        # For future use
        action = None

        # Handle linear region counting
        if len(self.region) == 2:
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    if current_centroid[0] > prev_position[0]:
                        action = "IN"
                    else:
                        action = "OUT"
                else:
                    if current_centroid[1] > prev_position[1]:
                        action = "IN"
                    else:
                        action = "OUT"
                self.counted_ids.append(track_id)

        # Handle polygonal region counting
        elif len(self.region) > 2:
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                if current_centroid[0] > prev_position[0]:
                    action = "IN"
                else:
                    action = "OUT"
                self.counted_ids.append(track_id)

    def display_counts(self, im0):
        """Display the counts and actions on the image."""
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
                                  f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

        for track_id in self.track_ids:
            track_index = self.track_ids.index(track_id)
            cls = self.clss[track_index]
            class_color = colors(int(cls), True)

            speed_label = f"{int(self.spd[track_id])} mph" if track_id in self.spd else self.names[int(cls)]
            combine_label = f"{self.names[int(cls)]}, {speed_label}, ID: {track_id}"
            self.annotator.box_label(self.boxes[self.track_ids.index(track_id)], label=combine_label, color=class_color)

    def count(self, im0):
        """Main counting function to track objects and store counts in the file."""
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]
            if track_id not in self.trk_pa:
                self.trk_pa[track_id] = 1

            speed_label = f"{int(self.spd[track_id])} mph" if track_id in self.spd else self.names[int(cls)]
            self.annotator.draw_centroid_and_tracks(self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width)

            current_area = (box[2] - box[0]) * (box[3] - box[1])
            
            # Always update speed estimation, regardless of intersection with line
            time_difference = time() - self.trk_pt.get(track_id, time())
            if time_difference > 0:
                distance_moved = current_area / self.trk_pa[track_id]
                if distance_moved < 1.035 and distance_moved > 0.965:
                    distance_moved = 0
                distance_moved *= 3
                self.spd[track_id] = distance_moved / time_difference  # Percentage area change per second

            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]
            self.trk_pa[track_id] = current_area

            self.count_objects(current_centroid, track_id, prev_position, cls)

        self.display_counts(im0)

        # Initialize writer with frame dimensions
        height, width, _ = im0.shape
        self.initialize_writer(width, height)

        # Write frame to video
        self.video_writer.write(im0)

        return im0
    
    def close(self):
        """Release the video writer."""
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved as {self.output_filename}")