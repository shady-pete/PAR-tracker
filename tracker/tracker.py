import argparse
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from utils.projpoint import projecter
from ultralytics.utils.ops import xyxy2xywh
from utils.parmodel import parmodel
from math import dist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import utils.save_person_state as save_person_state
import os
import torch

np.seterr(all=None, divide=None, over=None, under=None, invalid=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"System using {device}")
# Frame che devono passare tra crossing della linea da parte di una stessa persona
# per essere contati (dato che il punto considerato si sposta a causa del cambio di forma del bounding box,
# serve per evitare il conteggio di falsi crossing)
CROSS_COOLDOWN = 15
SKIP_FRAMES = 3 # Frames we skip
CLASSIFY_ALL = SKIP_FRAMES * 3 # How many frames we wait to classify again
ASPECT_RATIO = 1.5 # Minimum aspect ratio
MIN_DIST_THRESHOLD = 380  # Minimum distance for bbox re-id in px
MIN_HEIGHT = 40  # Min height to consider valid a bbox px
USE_MASK = False
MARGINS= (0.05, 0.05) # up and down

class LineTracker:
    def __init__(self, lines, parmodel=None):
        """
        Initialize the object tracker with line crossing detection

        :param lines: dictionary with lines_id as key, and line starting point,ending point and perpendicular point as values
        :param parmodel: model for PAR
        """
        if USE_MASK:
            self.model = YOLO('./model/yolov8m-seg.pt')
        else:
            self.model = YOLO('./model/yolov8m.pt')
        self.model.to(device)

        self.parmodel = parmodel

        # Tracking parameters
        self.tracking_params = {
            'conf': 0.25,  # Confidence threshold for detections
            'iou': 0.45,  # IoU threshold for tracking associations.
            'imgsz': (544, 960),  # Image size dimensions for processing.
            'tracker': './config/botsort.yaml', # Path to the configuration file for the tracking algorithm
            'classes': [0],  # List of class IDs to be tracked (0 is person).
            'max_det': 50,  # Maximum number of detections allowed per frame.
            'stream_buffer': True,  # Enable/disable buffering for streaming input (True = enabled).
            'verbose': False  # Enable/disable verbose logging (False = no detailed logs).
        }
        self.framecount = 0
        self.lines = lines
        self.in_scene_counter = 0
        self.person_states = {}
        self.cross_cooldowns = {}
        self.line_passage_count = {}
        self.bags = {}
        self.reverse_mask = None
        self.first_frame = None
        self.debug = False
        self.polygon = None

    def _update_cooldowns(self):
        """
        Update line crossing cooldown, used to avoid to count more than one crossing of a line caused by bounding boxes shape changes.
        """
        for key in list(self.cross_cooldowns.keys()):
            # If exists a cooldown
            if self.cross_cooldowns[key] > 0:
                # Decrese the counter by 1
                self.cross_cooldowns[key] -= 1
            else:
                # If the cooldown is expired, remove the key from the dict
                del self.cross_cooldowns[key]

    def _draw_lines(self, frame):
        """
        Draws the lines on the frame
        """
        # Draw the POLYGON
        if self.debug:
            for index in range(len(self.polygon.exterior.xy[0])-1):
                pt1= ( int( self.polygon.exterior.xy[0][index] ), int( self.polygon.exterior.xy[1][index] ) )
                pt2= ( int( self.polygon.exterior.xy[0][index+1] ), int( self.polygon.exterior.xy[1][index+1] ) )
                cv.line(frame, pt1, pt2, (0,255,0), 2)

        # Draw lines from the self.lines dict
        for key, points in self.lines.items():
            start_point, end_point, arrow_s, arrow_e = points

            # line start l.s, line end l.e
            lsx, lsy = start_point
            lex, ley = end_point
            line_color = (0, 255, 0)  # Green
            thickness = 3

            # Draw the line
            cv.line(frame, start_point, end_point, line_color, thickness)

            # Put the number on the top left
            if lsx >= lex:
                cv.putText(frame, str(key), (lex - 20, ley), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 3)
            else:
                cv.putText(frame, str(key), (lsx - 20, lsy), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 3)

            cv.arrowedLine(frame, arrow_s, arrow_e, line_color, thickness, tipLength=0.2)

    def check_crossed(self, prev_bbox, curr_bbox, person_id):
        """
        Checks for line crossing of a person, using line interpolation
        Returns list of id of line crossed (actually for debugging reasons it returns [(3,'right_direction),(4,'right_direction')])
        """
        crossed_lines = []
        for key, points in self.lines.items():
            # Retrieve the start and end points of the line, other than the direction arrow points
            line_start, line_end, arrow_s, arrow_e = points

            # Compute the width and height of the bbox
            _, _, w, h = xyxy2xywh(curr_bbox)
            if h / w < ASPECT_RATIO:  # if the aspect ratio is scuffed, it's likely that the feet of the person are not in the bb so ignore
                continue

            # dir is for debugging purposes
            crossed = self.is_line_crossed(line_start, line_end, self._bb2centerbottom(prev_bbox),
                                           self._bb2centerbottom(curr_bbox), arrow_s, arrow_e)
            # If the line is crossed and the person is not in cooldown, we count the crossing
            if crossed:
                if (person_id, key) not in self.cross_cooldowns:
                    crossed_lines.append(key)
                    self.cross_cooldowns[(person_id, key)] = CROSS_COOLDOWN
                else:  # Otherwise, we ignore the crossing because it was on cooldown
                    continue

                if key not in self.line_passage_count:  # update how many time the line has been crossed
                    self.line_passage_count[key] = 1
                else:
                    self.line_passage_count[key] += 1

        return crossed_lines

    def update_person_info(self, gender_probs, bag_probs, hat_probs, ids):
        person = self.person_states[ids]

        # If the predicted label is 0, add the probability, else it subtract the probability

        person['gender_probs'] += gender_probs[1] if gender_probs[0] == 0 else -gender_probs[1]
        person['bag_probs'] += bag_probs[1] if bag_probs[0] == 0 else -bag_probs[1]
        person['hat_probs'] += hat_probs[1] if hat_probs[0] == 0 else -hat_probs[1]

        # If probability > 0 classify as '0' class.
            # In case of gender 0 equals to 'M'
            # In case of hat 0 equals to 'no_hat'
            # In case of bag 0 equals to 'no_bag'
        # Else classifies as '1' class
        person['gender'] = 'M' if person['gender_probs'] >= 0 else 'F'
        person['hat'] = False if person['hat_probs'] >= 0 else True
        person['bag'] = False if person['bag_probs'] >= 0 else True

        person['classificazioni'] += 1

    def classify_all(self, frame):
        """
        Sends a list of all bounding box patches to the parmodel for classification and then updates the classification of gender,bag and hat
        """
        patches = []
        ids = []

        for key, person in self.person_states.items():
            # Extract bbox coordinates
            _, _, w, h = xyxy2xywh(person['bbox'])

            # If the person's bounding box respect a certain ASPECT_RATIO and the person is visible in the frame (in_scene)
            if h / w > ASPECT_RATIO and person['in_scene']:
                ids.append(key)
                # Extract the image patch for classification, depending on USE_MASK, uses or not the 'mask'
                if USE_MASK:
                    patches.append(self.extract_segmented_object(frame, person['bbox'], person['mask']))
                else:
                    patches.append(self.extract_patch(frame, person['bbox']))

        if patches:
            # Send patches to the classification model and takes account of the classification
            gender_probs, bag_probs, hat_probs = self.parmodel.classify(patches)
            for i in range(len(ids)):
                self.update_person_info(gender_probs[i], bag_probs[i], hat_probs[i], ids[i])

    def classify_single(self, frame, id):
        """
        Given id, call the parmodel to classify the bounding box and updates person_state
        """
        # Retrieve the person's state using their ID
        person = self.person_states[id]
        # Extract the image patch (corrisponding to the bbox) from the video frame
        patch = self.extract_patch(frame, person['bbox'])
        # Classify the patch, extracting probabilities
        gender_probs, bag_probs, hat_probs = self.parmodel.classify([patch])
        # Update the person's information
        self.update_person_info(gender_probs[0], bag_probs[0], hat_probs[0], id)

    def is_line_crossed(self, line_start, line_end, s_start, s_end, arrow_s, arrow_e):
        """
        Checks if the vector defined by s_start and s_end intersecate the vector defined by line_start and line_end
        then checks if the vector defined by s_start and s_end points in the same direction of the vector defined by arrow_s and arrow_e
        Implementation of https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        """

        def cross2d(x,
                    y):  # in np.cross --> Dimension-2 input arrays were deprecated in 2.0.0. If you do need this functionality, you can use:
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

        # Computer the vector representing the line segment
        line_vector = np.array(line_end) - np.array(line_start)
        # Compute the vector representing the movement
        movement_vector = np.array(s_end) - np.array(s_start)

        # Compute the vector representing the direction of the arrow
        arrow_vector = np.array(arrow_e) - np.array(arrow_s)
        # Compute the vector from the start of the moving segment to the start of the line segment.
        vector_start = np.array(s_start) - np.array(line_start)

        # Calculate the denominator `d` which determines if the two vectors are parallel or not.
        d = cross2d(line_vector, movement_vector)
        # If 'd == 0' the two vectors are parallel, and there can't be an intersection
        if d == 0:
            return False

        # Compute the scalar `t`, which determines how far along the `line_vector` the intersection occurs.
        # Compute the scalar `u`, which determines how far along the `movement_vector` the intersection occurs.
        t = cross2d(vector_start, movement_vector) / d
        u = - cross2d(line_vector, vector_start) / d

        # If t and u are between 0 and 1 it means there is an intersection (we are not interested in the specific point)
        # The second condition check if the the movement vector and arrow vector point in the same direction
        return (0 <= t <= 1 and 0 <= u <= 1) and (np.dot(arrow_vector, movement_vector) >= 0)

    def _bb2centerbottom(self, bb):
        """
        Given a bounding box, it return the point that is the bottom center point
        """
        # Extract bbox coordinates as Integers
        x1, _, x2, y2 = map(int, bb)
        # Calculate the center of the x-coordinate of the bottom edge
        x, y = ((x1 + x2) // 2, y2)
        # Return a tuple (x, y) which represents the bottom-center point
        return (x, y)

    def draw_text(self, img, text,
                  font=cv.FONT_HERSHEY_PLAIN,
                  pos=(0, 0),
                  font_scale=2,
                  font_thickness=2,
                  text_color=(0, 0, 255),
                  text_color_bg=(255, 255, 255)
                  ):
        # copied from: https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
        x, y = pos
        text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

        return text_size

    def display_person_info(self, frame, person_state):
        """
        Given the person_state, it displays all the info required on the frame
        """
        # BGR
        color = (0, 0, 255)

        # Extract the bbox coordinates and convert them to integers
        x1, y1, x2, y2 = map(int, person_state['bbox'])
        # Draw a bbox around the person
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # debug code: cerchio considerato per crossing
        if self.debug:
            cv.circle(frame, self._bb2centerbottom(person_state['bbox']), 3, 1, 3)

        # Display the person's unique ID
        label = str(person_state['person_id'])
        self.draw_text(frame, label, pos=(x1, y1))

        font_scale = 1
        font_thickness = 1
        text_color = (0, 0, 0)

        # Initialize vertical offset for multiple lines printing
        size = 0
        # Display the person's gender below bounding box
        label_gender = f"Gender: {person_state['gender']}"
        text_size = self.draw_text(frame, label_gender, pos=(x1, y2), font_scale=font_scale,
                                   font_thickness=font_thickness, text_color=text_color)
        size += (text_size[1] +3)

        # Display information whether the person is wearing a bag or a hat
        label_hb = ''
        if not  person_state['bag'] and not person_state['hat']:
            label_hb = 'No Bag No Hat'
        elif person_state['bag'] and person_state['hat']:
            label_hb = 'Bag Hat'
        elif person_state['bag'] and not person_state['hat']:
            label_hb = 'Bag'
        elif person_state['hat'] and not person_state['bag']:
            label_hb = 'Hat'

        text_size = self.draw_text(frame, label_hb, pos=(x1, y2 + size), font_scale=font_scale,
                                   font_thickness=font_thickness, text_color=text_color)
        size += (text_size[1]+3)

        # Display the person trajectory and which lines he crosses
        label_trajs = '[]'
        if person_state['trajectory']:
            label_trajs = '[' + ','.join([str(item[0]) for item in person_state['trajectory']]) + ']'
        
        text_size = self.draw_text(frame, label_trajs, pos=(x1, y2 + (size)), font_scale=font_scale,
                                       font_thickness=font_thickness, text_color=text_color)

    def reverse_mask_update(self, mask):
        # Extract the contour
        contour = mask.xy[0]
        # Convert the contour points to integers in order to use them with OpenCV functions
        contour = contour.astype(np.int32)
        # Reshape the contour (-1) infers the number of points
        contour = contour.reshape(-1, 1, 2)
        # Add the contour onto the reverse mask (color = white)
        _ = cv.drawContours(self.reverse_mask, [contour], -1, (255, 255, 255), cv.FILLED)

    def extract_segmented_object(self, frame, box, mask):
        """
        Extract the segmented object, masking all the other person in the scene before extraction
        and substituing them with the background frame captured at the start
        """
        # Convert the bbox coordinates to integers for indexing the frame
        x_min, y_min, x_max, y_max = map(int, box)

        # Extract the contour points from the segmentation mask.
        contour = mask.xy[0].astype(np.int32).reshape(-1, 1, 2)

        # Invert the existing reverse mask, creating an area to preserve the object of interest.
        reverse = cv.bitwise_not(self.reverse_mask)
        _ = cv.drawContours(reverse, [contour], -1, (255, 255, 255), cv.FILLED)

        # Convert the mask to a 3-channel image to match the frame's color dimension
        mask3ch = cv.cvtColor(reverse, cv.COLOR_GRAY2BGR)
        isolated = cv.bitwise_and(mask3ch, frame)

        # Masks out everything except the object's area within the current frame.
        isolated_black_pixels = (isolated == [0, 0, 0]).all(axis=-1)
        # Restore the background on the black pixels
        isolated[isolated_black_pixels] = self.first_frame[isolated_black_pixels]

        # Extract the region of interest defined by the bbox
        patch = isolated[y_min:y_max, x_min:x_max]

        return patch

    def extract_patch(self, frame, bbox):
        """
        Given the frame, the bounding box and the mask, it extracts the image that is contained by the bounding box
        """
        # Convert the bbox coordinates to integers for indexing the frame
        x1, y1, x2, y2 = map(int, bbox)
        # Extract the region of interest (patch) from the frame using the bounding box coordinates.
        patch = frame[y1:y2, x1:x2]
        # Return the extracted patch
        return patch

    def _draw_info(self, frame):
        """
        Displays how many people are in the scene and how many crossing have been done for each line
        """
        # Shows in the top-left corner how many people are in scene
        label_person = f"Total people {self.in_scene_counter}"
        off = self.draw_text(frame, label_person, pos=(0, 0), text_color=(0, 0, 0))
        # Store the size of the previous text for positioning subsequent text below it
        size = off[1]

        # Iterate through each line key and its corresponding crossing
        for key, crossing in self.line_passage_count.items():
            # Label for number of crossings for the current line
            label_c = f"Passages for line {key}: {crossing}"
            # Draw the label for this line below the previous label, incrementing the size of the previous label (3 pixel spacing)
            off = self.draw_text(frame, label_c, pos=(0, 0 + size + 3), text_color=(0, 0, 0))
            # Update the label offset cumulative height
            size += off[1]

        # return the updated frame with drawn information
        return frame

    def verify_ID(self, curr_id):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, self.person_states[curr_id]['bbox'])
        # Compute the center of the bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # Initialize the closest distance respect to 'None' person as infinite
        distance = (float('inf'), None)

        # Check is the center is within the Polygon
        if self.polygon.contains(Point(center[0], center[1])):
            # Iterate through all tracked person for a match based on distance
            for key, person in self.person_states.items():
                # Consider only the person who actually are not visible
                if not person['in_scene']:
                    x1, y1, x2, y2 = map(int, person['bbox'])
                    center_person = ((x1 + x2) // 2, (y1 + y2) // 2)
                    curr_dist = dist(center, center_person)
                    # Find the person nearest to the new bounding box, in order to re assign her the previous id
                    distance = (curr_dist, key) if curr_dist < distance[0] else distance

            # If the distance is smaller than the minimum distance threshold, update new person information
            if distance[0] < MIN_DIST_THRESHOLD:
                key = distance[1]
                for value in ['person_id', 'trajectory', 'hat_probs', 'bag_probs', 'gender_probs', 'classificazioni']:
                    self.person_states[curr_id][value] = self.person_states[key][value]
                # after the copy, delete the matched person state from the dict
                del self.person_states[key]

    def process_frame(self, frame):
        """Process a single frame with line crossing detection and PAR classification"""
        # Perform tracking with our parameters, persist keeps the tracking consistent across frames, processing done on GPU
        results = self.model.track(
            frame,
            **self.tracking_params,
            persist=True,
            device=device
        )
        self._update_cooldowns()

        if USE_MASK:
            # Initialize a blank reverse mask with the same dimensions as the frame
            self.reverse_mask = np.zeros(frame.shape[:2], np.uint8)

        # Process all detected objects in the current frame using the tracking results.
        self._process_detected_objects(frame, results)

        # For ensuring batch classification to happen periodically, classify at regular intervals of persons
        if self.framecount % CLASSIFY_ALL == 0:
            self.classify_all(frame)

        # Update the display of information
        self._update_display(frame)
        # Reset the counter for the number of people currently visible in the scene for the next frame.
        self.in_scene_counter = 0
        return frame

    def _process_detected_objects(self, frame, results):
        """Process all detected objects in the frame"""
        # If there are no bboxes, exit the function
        if not results[0].boxes:
            return

        # Extract bbox from results, eventually also the masks
        boxes = results[0].boxes
        masks = results[0].masks if USE_MASK else None

        # Init a list to keep track of IDs of objects
        ids_to_process = []

        # For each bbox
        for idx, box in enumerate(boxes):
            # Extract his ID
            id = int(box.id) if box.id is not None else None

            #If the ID already exists
            if id in self.person_states:
                # Process the person immediately
                self._process_person(frame, box, masks[idx] if masks else None)
            else:
                # Else, add it along with his index to the processing list
                ids_to_process.append((idx, box))

        # Process all new detections
        for _ in range(len(ids_to_process)):
            idx, box = ids_to_process.pop()
            self._process_person(frame, box, masks[idx] if masks else None)

    def _process_person(self, frame, box, mask=None):
        """Process a detected person"""
        # Extract the person ID, if not available set it to None
        person_id = int(box.id) if box.id is not None else None
        if not person_id:
            return

        # Converts the bbox coordinates from a PyTorch tensor to a numpy array
        bbox = box.xyxy[0].cpu().numpy()

        # If the person is already tracked
        if person_id in self.person_states:
            # Update it
            self._update_person_state(person_id, bbox, mask)
        else:
            # Else if the person wasn't in the dict, verify if the bbox is valid
            if not self._is_valid_person(bbox):
                return
            # Initialize a new state for the person
            self._initialize_person(person_id, bbox, mask)
            # Verify the ID
            self.verify_ID(person_id)
            # Classify the specific person
            self.classify_single(frame, person_id)

    def _is_valid_person(self, bbox):
        """Check if detected person meets valid criteria"""
        # Extract coordinates from the bounding box (x1, y1 for the top-left, x2, y2 for the bottom-right)
        x1, y1, x2, y2 = map(int, bbox)
        # Calculate the center point of the bbox
        center_point = Point((x1 + x2) // 2, (y1 + y2) // 2)
        # Check if the center falls inside a polygon or bbox meets a minimum threshold
        return self.polygon.contains(center_point) or (y2 - y1) >= MIN_HEIGHT

    def _initialize_person(self, person_id, bbox, mask=None):
        """Initialize a new person state"""
        self.person_states[person_id] = {
            'person_id': person_id,
            'previous_bbox': bbox,
            'bbox': bbox,
            'trajectory': [],
            'hat': False,
            'hat_probs': 0,
            'bag': False,
            'bag_probs': 0,
            'gender': 'M',
            'gender_probs': 0,
            'in_scene': True,
            'classificazioni': 0
        }
        # If we are using a mask, add it to the person_states dictionary
        if mask is not None:
            self.person_states[person_id]['mask'] = mask

    def _update_person_state(self, person_id, bbox, mask=None):
        """Update existing person state"""

        # Retrieve the current state of the person using their ID
        person_state = self.person_states[person_id]
        
        # Mark the person as "in_scene", indicating they are currently detected
        person_state['in_scene'] = True
        
        # Update the current bounding box of the person
        person_state['bbox'] = bbox
        
        # If we are using a mask, update it and reverse it
        if mask is not None:
            person_state['mask'] = mask
            self.reverse_mask_update(mask)

        # Increase person counter
        self.in_scene_counter += 1

        # Checks for line crossings
        line_crossed = self.check_crossed(
            person_state['previous_bbox'],
            bbox,
            person_id
        )

        # If a line was crossed, append the information to the person trajectory
        if line_crossed:
            person_state['trajectory'].append(line_crossed)

        # Update the previous bounding box to the current one for next frame comparison
        person_state['previous_bbox'] = bbox

    def _update_display(self, frame):
        """Update frame display with person info and lines"""
        for person_state in self.person_states.values():
            if person_state['in_scene']:
                self.display_person_info(frame, person_state)
                person_state['in_scene'] = False

        self._draw_lines(frame)
        self._draw_info(frame)

    def process_video_classic(self, input_path, output_path):
        """
        Process an entire video for object tracking with line crossing

        :param input_path: Path to input video file
        :param output_path: Path to output video file
        """
        # Open input video
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {input_path}")
            return

        # Setup video writer
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter(output_path, fourcc, 25.0, (frame_width, frame_height))

        # Pause flag
        paused = False
        last_processed_frame = None

        # Choosing point for the forbidden zone
        self.polygon = Polygon([
            (int(0.01 * frame_width), int(MARGINS[0] * frame_height)),
            (int(0.01 * frame_width), int((1 - MARGINS[1]) * frame_height)),
            (int(0.99 * frame_width), int((1 - MARGINS[1]) * frame_height)),
            (int(0.99 * frame_width), int(MARGINS[0] * frame_height))
        ])
        
        # Process video frames
        while cap.isOpened():
            # Handle pause functionality
            key = cv.waitKey(1)
            if key == ord('p') or key == ord('P'):
                paused = not paused

            # Exit condition
            if key == ord('q'):
                break

            # If paused, continue showing the last frame
            if paused:
                if last_processed_frame is not None:
                    cv.imshow('Tracking', last_processed_frame)
                    #cv.imshow('Tracking', cv.resize(last_processed_frame, (int(frame_width/1.5), int(frame_height/1.5))))
                continue

            # Regular processing when not paused
            ret, frame = cap.read()
            if self.first_frame is None:
                self.first_frame = frame
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.framecount += 1
            if self.framecount % SKIP_FRAMES != 0:
                continue

            # Process and track objects in frame and store the last processed frame
            last_processed_frame = processed_frame = self.process_frame(frame)

            # Write frame to output video
            out.write(processed_frame)

            # Display frame (optional)
            cv.imshow('Tracking', processed_frame)
            #cv.imshow('Tracking', cv.resize(processed_frame, (int(frame_width / 1.5), int(frame_height / 1.5))))

        # Releases resources at the end (closes video files and the writer)
        cap.release()
        out.release()
        cv.destroyAllWindows()

        return self.person_states

    def show_video(self, input_path):
        # Opens a video file from the specified input
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():  # Checks if the video was opened successfully
            print(f"Error: Cannot open video file {input_path}")
            return

        # Continues as long as the video is open and there are frames to read
        while cap.isOpened():
            success, frame = cap.read()  # Reads a single frame from the video
            if not success:  # If no more frames are available, exits the loop
                break

            # Saves the first frame read
            if self.first_frame is None:
                self.first_frame = frame

            # Skips frames according to the SKIP_FRAMES parameter
            if self.framecount % SKIP_FRAMES == 0:
                processed_frame = self.process_frame(frame)  # Processes the frame

            # Displays the processed frame in a window
            cv.imshow('Tracking', processed_frame)
            cv.waitKey(1)

            # Increments the counter of processed frames
            self.framecount += 1

        # Releases the resources used by the video
        cap.release()

        # Returns the information about the state of the people processed in the video
        return self.person_states
    



if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process video with person tracking and line crossing detection.')

    parser.add_argument('--input', type=str, default="../video.mp4",
                      help='Path to the input video file (default: ../video.mp4)')
    
    parser.add_argument('--output', type=str, default="./output/output.mp4",
                      help='Path to the output video file (default: ./output/output.mp4)')
    
    parser.add_argument('--config', type=str, default='./config/camera.json',
                      help='Path to the camera configuration file (default: ./config/camera.json)')

    
    args = parser.parse_args()

    # Reads the camera setup and loads the points
    projecter = projecter(args.config)
    lines = projecter.get_projected_points_with_perpendiculars()




    # Model loading
    modelpar = parmodel('./net/par_model.pth')

    # Initializes the tracker instance to track people and identify line crossings
    tracker = LineTracker(
        lines=lines,
        parmodel=modelpar
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Starts video processing
    person_states = tracker.process_video_classic(args.input, args.output)

    # Save output
    output_json = os.path.join(os.path.dirname(args.output), "output.json")
    save_person_state.save(person_states, output_json)