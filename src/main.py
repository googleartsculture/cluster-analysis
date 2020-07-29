#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import collections
import copy
import logging
import os
import re
from functools import wraps
from io import BytesIO

import networkx as nx
import numpy
from flask import Flask, json, jsonify, request
from flask_cors import CORS
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from werkzeug.exceptions import (BadRequest, Forbidden, HTTPException,
                                    InternalServerError)

from models.polygon import Polygon

# Define which port to run on.
PORT = os.environ.get('CLUSTER_SERVICE_PORT', '8080')

# Should we be in debug mode
DEBUG = os.environ.get('DEBUG', True)
if DEBUG == 'False' or DEBUG == '0':
    DEBUG = False

# Pixels limit
pixel_limit = 50000
# Threshold value for DBSCAN
#dbscan_threshold = 160
# DBSCAN eps
dbscan_eps = 2
# DBSCAN minimum samples
dbscan_min_samples = 3
# Minimum inner glyphs to qualify as a cartouche
cartouche_min_glyph_count = 4


# Define a new get_headers() function for the HTTPException class,
# to return application/json MIME type rather than plain HTML
# @TODO Review how this is being done and implement in a more pythoninc way
def get_headers(self, environ=None):
    return [('Content-Type', 'application/json')]


HTTPException.get_headers = get_headers


# Define a new get_body() function for the HTTPException class,
# to return json rather than plain HTML
# @TODO Review how this is being done and implement in a more pythoninc way
def get_body(self, environ=None):
    return json.dumps({
        'success': False,
        'message': self.description,
        'code': self.code
    })


HTTPException.get_body = get_body


def get_buffered_points(points, centroid=None, factor=2.0):
    """
    Add buffer to points in a plane. For example, to expand a convex hull
    param points: Points we want to buffer
    param centroid: Centroid of the points
    param factor: Defines scalar product for point vectors
    return numpy array of new point coordinates
    """

    if centroid is None:
        centroid = [0, 0]

    buffered_points = []
    if len(points) == 0:
        return numpy.array(buffered_points)
    centroid = numpy.array(centroid)
    for p in numpy.array(points):
        buffered_points.append(((p - centroid) * factor) + centroid)
    buffered_points = [[numpy.rint(p[0]), numpy.rint(p[1])]
                        for p in buffered_points]
    return numpy.array(buffered_points)


# Create a wrapper for ensuring API requests have an
# application/JSON MIME type, raising a custom BadRequest if they don't
# @TODO Review whether there is a better way to implement this.
# A decorator seems like the most sensible, but mayber there is a better way?
def require_json(params=None):
    '''Decorator function to wrap app route functions when we explicity want
    the Content_Type to be application/json. Checks the request.is_json
    and raises a BadRequest exception if not. Also allows for a list of
    required parameters and checks for their existence, rasing a BadrRequest if
    any of them are missing.'''
    if params is None:
        params = []

    def require_json_inner(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            if request.is_json:
                json_payload = request.get_json(cache=False)
                for param in params:
                    if not json_payload.get(param):
                        raise BadRequest(
                            'The request is missing the {} parameter'.format(
                                param))

                return func(json_payload, *args, **kwargs)

            raise BadRequest(
                'The request content type must be application/json')

        return func_wrapper

    return require_json_inner


def do_cluster_analysis(image, direction, dbscan_threshold=160):

    job_dict = {
        'clusters_raw': {},
        'clusters_processed': {},
        'bounding_boxes': [],
        'groups': [],
        'cartouches': [],
        'direction': direction
    }

    # Process our source image to extract pixels of interest
    pixels, image_size = get_source_image_pixels(image,
                                                    threshold=dbscan_threshold)
    if len(pixels) == 0:
        raise BadRequest(
            'No pixels within the threshold range, please try a higher' +
            ' contrast image.')

    if len(pixels) >= pixel_limit:
        raise BadRequest(
            'Pixel limit of {} was exceeded. Please refine the edge detection'
            + ' or use a smaller segment.'.format(pixel_limit))

    # Do the clustering on our thresholded pixel coordinates then
    # check to see if we have found any clusters
    db_attributes = DBSCAN(eps=dbscan_eps,
                            min_samples=dbscan_min_samples).fit(pixels)
    if not have_clusters(db_attributes):
        raise BadRequest('No clusters were found.')

    # Create a dictionary of our raw clusters and a copy for the
    # clusters we are going to process
    job_dict['clusters_raw'] = get_clusters_dictionary(pixels, db_attributes)
    job_dict['clusters_processed'] = copy.deepcopy(job_dict['clusters_raw'])

    # Cartouche identification and merging
    # 1. Build the merge dict
    # 2. Find cartouches
    # 3. Merge clusters in the merge dict
    merge_dict = get_merge_dict(job_dict)
    cartouche_list = get_cartouches(merge_dict,
                                    min_glyph_count=cartouche_min_glyph_count)
    job_dict['cartouches'] = cartouche_list
    merge(job_dict, merge_dict, cartouche_list)
    set_groups_and_sequence(job_dict, labels_axis=0, direction=direction)

    # Restructure response...
    # Going forward, this will be changes and we will simply generate the
    # response in the correct format, but this is the quickest way to achieve
    # this for now
    response = {
        'clusters': [],
        'direction': direction,
        'cartouches': job_dict['cartouches'],
        'groups': job_dict['groups']
    }
    for i, sequence_key in enumerate(job_dict['sequence']):
        #raw_cluster = job_dict['clusters_raw'][sequence_key]
        processed_cluster = job_dict['clusters_processed'][sequence_key]
        cluster_vertices = []
        if len(set([p[0] for p in processed_cluster])) == 1 or len(
                set([p[1] for p in processed_cluster])) == 1:
            box_vertices = [
                list(vertex)
                for vertex in get_box_vertices(bounding_box(processed_cluster))
            ]
            for _, vertex in enumerate(box_vertices):
                cluster_vertices.append({
                    'x': int(vertex[0]),
                    'y': int(vertex[1])
                })
        else:
            # Get the convex hull and gather the coordinates of the vertices
            hull = ConvexHull(numpy.array(processed_cluster))
            centroid = [
                numpy.mean(hull.points[hull.vertices, 0]),
                numpy.mean(hull.points[hull.vertices, 1])
            ]
            buffered_points = get_buffered_points(hull.points[hull.vertices],
                                                    centroid=centroid,
                                                    factor=1.1)
            for _, point in enumerate(buffered_points):
                cluster_vertices.append({'x': point[0], 'y': point[1]})
        box = [b.tolist() for b in bounding_box(processed_cluster)]
        #raw = [{'x': r[0], 'y': r[1]} for r in raw_cluster]
        #processed = [{'x': r[0], 'y': r[1]} for r in processed_cluster]
        bounds = {
            'x': box[0],
            'y': box[1],
            'width': box[2] - box[0],
            'height': box[3] - box[1],
            'order': i
        }
        # rbg 18/09/2018 now returning the convex hull of a cluster
        response['clusters'].append({
            'cluster_id': sequence_key,
            'hull': cluster_vertices,
            'bounds': bounds
        })

    return response


def get_source_image_pixels(image, threshold=160, invert=False):
    imagePixels = image.load()
    # Convert the RGB image array to a set of gray scale
    # pixel coordinates over a certain threshold
    # These pixel cooordinates define the feature space
    pixels = []
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            grayScale = (imagePixels[x, y][0] + imagePixels[x, y][1] +
                            imagePixels[x, y][2]) / 3
            if (grayScale < threshold):
                pixels.append([x, y])
    return (pixels, image.size)


def have_clusters(db_attributes):
    if len(db_attributes.labels_) == 0:
        return False

    else:
        if numpy.all(db_attributes.labels_ == -1):
            return False

    return True


def get_clusters_dictionary(pixels, db_attributes):
    """
    Put clusters into a dictionary where the key is the
    cluster label and the value is an array of cluster [x,y]
    coordinates
    param db_attributes: Attributes generated by DBSCAN fit() method
    return A dictionary of clusters
    """
    clusters_dict = {}
    for cluster_label in set(db_attributes.labels_):
        if cluster_label == -1:
            # Ignore noise
            continue
        cluster = []
        # Build a cluster
        cluster = [
            pixels[j]
            for j in numpy.where(db_attributes.labels_ == cluster_label)[0]
        ]
        # Store the cluster in the dictionary
        clusters_dict[str(cluster_label)] = cluster
    return clusters_dict


def bounding_box(cluster):
    box = []
    cluster = numpy.array(cluster)
    box.append(min(cluster[:, 0]))
    box.append(min(cluster[:, 1]))
    box.append(max(cluster[:, 0] + 1))
    box.append(max(cluster[:, 1] + 1))
    return box


def inside(box1, box2):
    return box2[0] >= box1[0] and box2[1] >= box1[1] and box2[2] <= box1[
        2] and box2[3] <= box1[3]


def poly_inside(cluster_1, cluster_2):
    """
    Check whether the polygon of cluster_2 is entirely
    inside the polygon of cluster_1
    param cluster_1: The outer cluster
    param cluster_2: The putativee inner rectangle
    return: Boolean, True if cluster_2 is inside cluster_1 else False
    """
    hull_outer = ConvexHull(numpy.array(cluster_1))
    points_outer = numpy.array(cluster_1)
    poly = Polygon(points_outer[hull_outer.vertices, 0],
                    points_outer[hull_outer.vertices, 1])
    points_inner = numpy.array(cluster_2)
    insiders = (poly.is_inside([point[0] for point in points_inner],
                                [point[1] for point in points_inner]) >= 0)
    return numpy.all(insiders == True)


def get_merge_dict(job_dict):
    """
    Get clusters to merge, i.e. clusters that are entirely
    within another cluster into the outer cluster.
    The dictionary key is a cluster identifier defining
    an outer cluster. A dictionary value is a list
    of inner cluster labels.
    param job_dict: A job's dictionary
    return: A dictionary defining clusters to merge
    """
    # Compute the bounding box for all raw clusters
    boxes = {}
    for cluster_label in job_dict['clusters_raw']:
        boxes[cluster_label] = bounding_box(
            job_dict['clusters_raw'][cluster_label])
    # Look for clusters within other clusters based on bounding boxes
    merge_dict = {}
    for i in boxes:
        for j in boxes:
            if i == j:
                continue
            else:
                if inside(boxes[i], boxes[j]):
                    # Check that inner polygon is entirely inside outer polygon
                    if poly_inside(job_dict['clusters_raw'][i],
                                    job_dict['clusters_raw'][j]):
                        if i in merge_dict:
                            merge_dict[i].append(j)
                        else:
                            merge_dict[i] = [j]
    # Rationalise the merge dict so that values
    # contain only clusters that are one level down from the
    # the outer cluster
    for k in merge_dict:
        for c in merge_dict[k]:
            if c in merge_dict:
                merge_dict[k] = [
                    x for x in merge_dict[k] if x not in merge_dict[c]
                ]
    return merge_dict


def get_cartouches(merge_dict, min_glyph_count=3):
    """
    Use the merge dictionary (see get_merge_dict() function)
    to find cartouches.
    Generates a list of cluster dictionaries where a
    cluster dictionary has the following keys:
    - inner a list of clusters labels that are the glyph candidates
    within a cartouche frame
    - wrapper a list containing a single cluster label which is
    the wrapper for the inner glyphs
    - frames a list containing cluster labels for any outer frames
    param merge_dict: A dictionary defining clusters
    param min_glyph_count: Defines the minimum number of glyphs in a cartouche
    return: A list of cartouche dictionaries
    """
    cartouche_list = []
    # If the merge_dict is empty then there can't be any cartouches
    if len(merge_dict) == 0:
        return cartouche_list
    # Generate the edge list based on merge dict
    elist = []
    for k in merge_dict:
        for cluster in merge_dict[k]:
            t = (k, cluster)
            elist.append(t)
    # Create a directed graph
    DG = nx.DiGraph()
    DG.add_edges_from(elist)
    # Iterate nodes from the bottom
    for i, node in enumerate(reversed(list(DG.nodes))):
        cartouche_dict = {}
        successors = list(DG.successors(node))
        if len(successors) >= min_glyph_count:
            is_cartouche = True
            cartouche_dict['inner'] = successors
            cartouche_dict['wrapper'] = [node]
            cartouche_dict['frames'] = sorted(list(nx.ancestors(DG, node)))
            if len(cartouche_dict['frames']) == 0:
                cartouche_list.append(cartouche_dict)
            else:
                for cluster in cartouche_dict['frames']:
                    successors = list(DG.successors(cluster))
                    if len(successors) > 1:
                        is_cartouche = False
                        break
                if is_cartouche:
                    cartouche_list.append(cartouche_dict)
    return cartouche_list


def merge(job_dict, merge_dict, cartouche_list):
    """
    Use the merge dictionary and cartouche list to merge clusters.
    Cartouche inner clusters are not merged with
    their frames but merging is supported for clusrers
    within the the cartouche inner clusters
    param job_dict: A job's dictionary
    param merge_dict: A dictionary defining clusters
    param cartouche_list: A list of cartouche dictionaries
    """

    # Build a list of cartouche wrappers and frames so
    # we don't merge cartouche inner clusters with their frames
    cartouche_clusters = []
    for c in cartouche_list:
        cartouche_clusters = cartouche_clusters + c['wrapper']
        cartouche_clusters = cartouche_clusters + c['frames']
    # Store the labels of inner clusters that we merge with outer clusters
    pop_list = []
    for outer in merge_dict:
        if outer not in cartouche_clusters:
            for inner in merge_dict[outer]:
                # We may have already processed our raw clusters so
                # want to avoid a key error
                if outer in job_dict[
                        'clusters_processed'] and inner in job_dict[
                            'clusters_processed']:
                    # Merge the 2 clusters and label with
                    # the name of the outer cluster then delete
                    # the inner cluster
                    job_dict['clusters_processed'][outer] = merge_clusters(
                        job_dict['clusters_processed'][inner],
                        job_dict['clusters_processed'][outer])
                    pop_list.append(inner)
    # Remove the clusters that have been merged
    for inner in pop_list:
        job_dict['clusters_processed'].pop(inner)


def merge_clusters(inner_cluster, outer_cluster):
    """
    Merge two clusters
    param inner_cluster: The cluster that's inside another cluster
    param outer_cluster: The cluster that's outside another cluster
    """
    outer_cluster = outer_cluster + [
        item for item in inner_cluster if item not in outer_cluster
    ]
    return outer_cluster


def set_groups_and_sequence(job_dict, labels_axis=0, direction='ltr'):
    clusters_dict = job_dict['clusters_processed']
    labels_dict = get_labels(clusters_dict,
                                labels_of_interest=[],
                                axis=labels_axis,
                                direction=direction)
    label_lists = list(labels_dict.values())
    # Build a dictionary of cluster bounding boxes
    box_dict = {}
    cluster_labels = [
        label for inner_list in label_lists for label in inner_list
    ]
    for cluster_label in cluster_labels:
        box_dict[cluster_label] = bounding_box(clusters_dict[cluster_label])
    # Start building the groups
    groups = []
    g = []
    # Iterate the labels list
    for i in range(1, len(label_lists)):
        if any(label in label_lists[i] for label in label_lists[i - 1]):
            # If the current label list shares any labels with
            # the previous labels list then add
            # these labels to our group if they aren't already
            # in the group
            g = g + list(set(label_lists[i]) - set(g))
            g = g + list(set(label_lists[i - 1]) - set(g))
        else:
            # We crossed a group boundary so add the group we've being building
            # to the groups list
            if len(g) > 0:
                # Sort within a group
                groups.append(order_group(g, box_dict, direction))
                g = []
    if len(g) > 0:
        # Sort within a group
        groups.append(order_group(g, box_dict, direction))
    job_dict['groups'] = groups
    job_dict['sequence'] = [
        cluster_label for group in groups for cluster_label in group
    ]


def y_overlaps(cluster1, cluster2):
    """
    author: Jon Liberal
    Test if there is y-axis overlap for 2 clusters
    param cluster1 First cluster's coordinates
    param cluster2 Second cluster's coordinates
    return Boolean: True if overlap else False
    """
    return not (cluster2[0] > cluster1[1] or cluster2[1] < cluster1[0])


def order_group(group, box_dict, read_order):
    """
    author: Jon Liberal
    Order clusters inside a group
    param group: List of cluster labels
    param box_dict: A dictionary of cluster bounding boxes
    param read_order: The read order for the glyphs
    return a list containing the group's clusters in correct read-order
    """
    # Generate a coordinate data structure from the bounding boxes
    candidates = {}
    for label in group:
        box = box_dict[label]
        if read_order == 'rtl' or read_order == 'vrtl':
            candidates[label] = ((box[0], 0 - box[3]), (box[2], 0 - box[1]))
        else:
            candidates[label] = ((0 - box[2], 0 - box[3]), (0 - box[0],
                                                            0 - box[1]))

    order = []
    while len(candidates) > 0:
        #Create some side lists to simplify code
        yDict = {}
        xDict = {}
        for rectangle in candidates:
            tuples = candidates[rectangle]
            yDict[rectangle] = (tuples[0][1], tuples[1][1])
        for rectangle in candidates:
            tuples = candidates[rectangle]
            xDict[rectangle] = (tuples[0][0], tuples[1][0])
        #A
        rightCand = candidates
        while len(rightCand) > 0:
            #1st step
            yres = -100000
            for rectangle in rightCand:
                if yDict[rectangle][1] >= yres:
                    yres = yDict[rectangle][1]
                    topRectangle = rectangle
            #2nd step
            auxRightCand = []
            xres = xDict[topRectangle][1]
            for rectangle in rightCand:
                if y_overlaps(
                        yDict[rectangle],
                        yDict[topRectangle]) and xDict[rectangle][0] >= xres:
                    auxRightCand.append(rectangle)
            rightCand = auxRightCand
        #B
        order.append(topRectangle)
        candidates.pop(topRectangle)
    return order


def get_labels(clusters_dict, labels_of_interest=[], axis=0, direction='ltr'):
    step = 1
    if axis == 0:
        # x-axis labelling
        box_min_index = 0
        box_max_index = 2
        if direction == 'rtl':
            # Go in reverse on the x-axis
            box_min_index = 2
            box_max_index = 0
            step = -1
    else:
        # y-axis labelling because we have a column and we always read from top
        # If we have a read-order of vrtl we sort by reversed
        # x values within groups
        box_min_index = 1
        box_max_index = 3
    d = {}
    if len(labels_of_interest) == 0:
        labels_of_interest = list(clusters_dict.keys())
    for cluster_label in labels_of_interest:
        box = bounding_box(clusters_dict[cluster_label])
        for i in range(box[box_min_index], box[box_max_index], step):
            if i not in d:
                d[i] = []
                d[i].append(cluster_label)
            # Append another cluster label only if there's no
            # overlap of the cluster's bounding
            # box with others in the label set.
            # One bounding box entirely within another is permitted
            else:
                if not have_overlaps(cluster_label, d[i], clusters_dict, axis):
                    d[i].append(cluster_label)
    od = collections.OrderedDict(sorted(d.items()))
    if direction == 'rtl' and axis == 0:
        od = collections.OrderedDict(reversed(list(od.items())))
    return od


def have_overlaps(label, labels_list, clusters_dict, axis=0):
    label_box = bounding_box(clusters_dict[label])
    for list_label in labels_list:
        if list_label is not label:
            list_label_box = bounding_box(clusters_dict[list_label])
            if not inside_axis(list_label_box, label_box, axis) \
                    and not inside(list_label_box, label_box)\
                    and overlaps(list_label_box, label_box):
                return True
    return False


def inside_axis(box1, box2, axis=0):
    if axis == 0:
        # Find the box with the maximum width on the x-axis
        box1_width = box1[2] - box1[0]
        box2_width = box2[2] - box2[0]
        if box1_width >= box2_width:
            # box1 is wider or the same width as box2
            if box2[0] >= box1[0] and box2[2] <= box1[2]:
                return True
            else:
                return False
        else:
            # box2 is wider than box1
            if box1[0] >= box2[0] and box1[2] <= box2[2]:
                return True
            else:
                return False
    else:
        # Find the box with the maximum height on the y-axis
        box1_height = box1[3] - box1[1]
        box2_height = box2[3] - box2[1]
        if box1_height >= box2_height:
            # box1 is higher or the same height as box2
            if box2[1] >= box1[1] and box2[3] <= box1[3]:
                return True
            else:
                return False
        else:
            # box2 is higher than box1
            if box1[1] >= box2[1] and box1[3] <= box2[3]:
                return True
            else:
                return False


def overlaps(box1, box2):
    for [b1, b2] in [[box1, box2], [box2, box1]]:
        v2 = get_box_vertices(b2)
        for vertex in v2:
            if ((b1[0] <= vertex[0]) and
                (vertex[0] <= b1[2])) and ((b1[1] <= vertex[1]) and
                                            (vertex[1] <= b1[3])):
                return True
    return False


def get_box_vertices(box):
    vertices = [[box[0], box[3]], [box[0], box[1]], [box[2], box[1]],
                [box[2], box[3]]]
    return numpy.array(vertices)


def find_plural_marks(job_dict, width_threshold, height_threshold):
    # Find processed cluster width, heights, and labels
    widths = []
    heights = []
    cluster_labels = []
    for cluster_label in job_dict['clusters_processed']:
        cluster = numpy.array(job_dict['clusters_processed'][cluster_label])
        widths.append(max(cluster[:, 0]) - min(cluster[:, 0]))
        heights.append(max(cluster[:, 1]) - min(cluster[:, 1]))
        cluster_labels.append(cluster_label)

    # Normalise widths and heights to percentages and generate a points array.
    # This way we get scale invariance
    points = [[w, h] for (w, h) in list(
        zip([int(w) for w in (widths / max(widths)) *
             100], [int(h) for h in (heights / max(heights)) * 100]))]

    # List clusters with a normalised width and height under 20%
    plural_marks = []
    for i in range(len(points)):
        if points[i][0] < width_threshold and points[i][1] < height_threshold:
            plural_marks.append(cluster_labels[i])
    return plural_marks


def serialize_job_dictionary(job_dict):
    file_name = "jobs" + os.path.sep + "dictionary.json"
    with open(file_name, 'w') as outfile:
        json.dump(job_dict, outfile)


app = Flask(__name__)
CORS(app)


@app.route('/_ah/warmup')
def warmup():
    return '', 200, {}


@app.route('/clusteranalysis', methods=['POST'])
@require_json(['image'])
def cluster_analysis(payload):
    """Cluster analysis endpoint to perform cluster
    analysis on the given image data"""

    # Create an image object from the image data and validate it
    try:
        image = payload.get('image')
        imagedata = re.sub('^data:image/.+;base64,', '',
                            image)  # Strip the image meta
        imagebytes = BytesIO(base64.b64decode(imagedata))
        image = Image.open(imagebytes)

    except base64.binascii.Error as e:
        logging.error(e)
        raise BadRequest('Unable to process image data: {}'.format(e))

    except TypeError as e:
        logging.error(e)
        message = e.message.message if isinstance(
            e.message, base64.binascii.Error) else e.message
        raise BadRequest('Unable to process image data: {}'.format(e.message))

    except IOError as e:
        logging.error(e)
        message = list(e.args)[0]
        raise BadRequest(message)

    if not isinstance(image, PngImageFile):
        raise BadRequest('Only png images are accepted')

    # Validate the read order
    direction = payload.get('direction', 'rtl')
    if direction not in ['rtl', 'vrtl', 'ltr', 'vrtl']:
        raise BadRequest("Invalid read order '{}'".format(direction))

    try:
        if 'threshold' in payload:
            threshold = payload.get('threshold')
            try:
                threshold = int(threshold)
                result = do_cluster_analysis(image,
                                                direction,
                                                dbscan_threshold=threshold)
            except ValueError:
                result = do_cluster_analysis(image, direction)
        else:
            result = do_cluster_analysis(image, direction)
        return jsonify(code=200, success=True, result=result)

    except Exception as e:
        logging.error(e)
        raise InternalServerError('Something went wrong!')


if __name__ == '__main__':
    # By default PORT is 8080, if you want to change it for local dev
    # set a CLUSTER_SERVICE_PORT environment variable.
    app.run(host='127.0.0.1', port=PORT, debug=DEBUG)
