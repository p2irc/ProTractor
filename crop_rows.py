#TODO refactor imports to match pep-8 style
import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import json
import pickle
import os
import math
import statistics
import skimage.transform
from shapely.geometry import box, LineString, Point
from klines import get_k_lines

def scale_image(img, old_min, old_max, new_min, new_max, rint=False, np_type=None):
    """
    :param img: numpy array
    :param old_min: scalar old minimum pixel value
    :param old_max: scalar old maximum pixel value.
    :param new_min: scalar new minimum pixel value
    :param new_max: scalar new maximum pixel value.
    :param rint: Should the resulting image be rounded to the nearest int values? Does not convert dtype.
    :param np_type: Optional new np datatype for the array, e.g. np.uint16. If none, keep current type.
    :return: scaled copy of img.
    """
    # equivalent to:
    # img = (new_max - new_min) * (img - old_min) / (old_max - old_min) + new_min
    # see https://stats.stackexchange.com/a/70808/71483 and its comments.

    a = (new_max - new_min) / (old_max - old_min)
    b = new_min - a * old_min
    # This autoconverts to float64, preventing over-/under-flow in most cases.
    img = a * img + b
    if rint:
        img = np.rint(img)
    if np_type:
        img = img.astype(np_type)
    return img

def scale_image_auto(img, new_min, new_max, rint=False, np_type=None):
    """
    :param img: numpy array
    :param new_min: scalar new minimum pixel value
    :param new_max: scalar new maximum pixel value.
    :param rint: Should the resulting image be rounded to the nearest int values? Does not convert dtype.
    :param np_type: Optional new np datatype for the array, e.g. np.uint16. If none, keep current type.
    :return: copy of img, with all pixels scaled according to the global max and min pixels (across all channels),
            tuple of (old_min, old_max).
    """
    if np.issubdtype(img.dtype, np.bool_):
        # scale_image() can't handle the bool type.
        img = np.uint8(img)
    # if this is an intermediate image, there could be nans.
    old_max = np.nanmax(img)
    old_min = np.nanmin(img)
    return scale_image(img, old_min, old_max, new_min, new_max, rint=rint, np_type=np_type)

def _get_half_angle_rad(angle_of_view_degrees):
    angle_of_view_rad = angle_of_view_degrees * math.pi / 180
    return angle_of_view_rad / 2


def __get_frame_size_at_distance(angle_of_view_degrees, distance_m):
    half_angle_rad = _get_half_angle_rad(angle_of_view_degrees)
    half_frame_size_m = math.tan(half_angle_rad) * distance_m
    return 2 * half_frame_size_m


def __get_distance_for_frame_size(angle_of_view_degrees, frame_size_m):
    half_angle_rad = _get_half_angle_rad(angle_of_view_degrees)
    half_frame_size_m = frame_size_m / 2
    distance_m = half_frame_size_m / math.tan(half_angle_rad)
    return distance_m

def get_mean_px_per_meter_at_distance(angle_of_view_degrees, distance_m, resolution_px):
    """
    angle_of_view_degrees and resolution_px must be along the same axis.
    """
    return resolution_px / __get_frame_size_at_distance(angle_of_view_degrees, distance_m)

def largest_indices(arr, n):
    """
    Returns the indices of the n largest elements.
    :param arr: A numpy array
    :param n: An positive integer
    :return: A numpy array of indices and shape
    """
    # Flattens nd-arrays, does nothing on a 1-d array
    flattened_array = arr.flatten()
    # Partition the array into the indices of largest values
    indices = np.argpartition(flattened_array, -n)[-n:]
    # Sorts in ascending order
    indices = indices[np.argsort(-flattened_array[indices])]
    return np.unravel_index(indices, arr.shape)


def warp_image(im, im_warped_shape, h_im_to_warped):
    """
    :param im: numpy array. Must be uint8, uint16, or float between -1 and 1.
    :param im_warped_shape: (height, width) of output image. Bands will be preserved automatically.
    :param h_im_to_warped: 3x3 numpy array, describing the homography transformation with which to warp the image.
    :return: image warped to the provided shape, using the provided transformation.
    """
    im_warped_height, im_warped_width = im_warped_shape[0:2]

    all_dims = np.array((*im.shape[0:2], *im_warped_shape[0:2]))
    if np.all(all_dims < 32766):
        im_warped = cv2.warpPerspective(src=im, M=h_im_to_warped, dsize=(im_warped_width, im_warped_height),
                                        flags=1)
    else:
        # slower but works for all image sizes.
        im_warped = skimage.transform.warp(im, inverse_map=np.linalg.inv(h_im_to_warped), output_shape=im_warped_shape,
                                           preserve_range=True)
        if np.issubdtype(im.dtype, np.integer):
            im_warped = np.rint(im_warped)
        im_warped = im_warped.astype(im.dtype)
    return im_warped


def get_homography_matrix(x, x_prime):
    """
    Let n be the number of corresponding coordinates.
    Let D be the dimensionality of both sets of coordinates.

    :param x: n-by-D array of coords.
    :param x_prime: n-by-D array of coords.

    x and x_prime must have an element-wise correspondence,
    so that x[0,:] and x_prime[0,:] refer to the same point,
    in different coordinate systems.

    :precond: n must be at least 4.

    NOTE: This function is only guaranteed to work for D = 2, but it may work for greater Ds.

    :return: The standard 3-by-3 homography matrix from x's coordinate system
            to x_prime's coordinate system, as determined using cv2.LMEDS.
    """

    # Note that cv2.getPerspectiveTransform is currently buggy, so findHomography should always be used:
    # https://github.com/opencv/opencv/issues/11944

    h, _ = cv2.findHomography(x, x_prime, method=cv2.LMEDS)
    return h


class Img(object):
    """Image object that will store the filename, image, and Row objects found in that image."""

    def __init__(self, filename):

        # Strip the path and extension from the filename
        self.filename = os.path.splitext(os.path.basename(filename))[0]
        self.path = filename
        # This param will never be modified
        self.bgr_img_initial = cv2.imread(filename)
        self.bgr_modified = cv2.imread(filename)
        self.Rows = []

    def set_img_modified(self, img_array):
        """Updates the image that may be modified"""
        self.bgr_modified = img_array

    def get_img_modified(self):
        """Return the image that may be modified"""
        return self.bgr_modified

    def get_img_initial(self):
        """Returns the initial unmodified image"""
        return self.bgr_img_initial

    def set_luminance_lab(self, luminance_adjust):
        """Sets the luminance value for an image to a set fixed value"""
        im_Lab = cv2.cvtColor(self.get_img_modified(), cv2.COLOR_BGR2LAB)
        im_L = im_Lab[:, :, 0]
        im_L[:, :] = luminance_adjust
        im_Lab = np.dstack((im_L, im_Lab[:, :, 1], im_Lab[:, :, 2]))
        self.set_img_modified(cv2.cvtColor(im_Lab, cv2.COLOR_LAB2BGR))

    def get_median_luminance_initial_img(self):
        """
        Returns the median luminance value (L channel in Lab space)
        for an image
        :return: int
        """
        im_lab = cv2.cvtColor(self.get_img_initial(), cv2.COLOR_BGR2LAB)
        im_l = im_lab[:, :, 0]
        return np.median(im_l.ravel())

    def apply_mask(self, mask):
        """
        Applies the passed in mask to the Img object's image property
        :param mask: Binary image of same dimensions as the Img object's initialization parameter
        """
        b_channel = np.multiply(self.get_img_modified()[:, :, 0], mask)
        g_channel = np.multiply(self.get_img_modified()[:, :, 1], mask)
        r_channel = np.multiply(self.get_img_modified()[:, :, 2], mask)
        # Update the image with the result
        self.set_img_modified(np.dstack((b_channel, g_channel, r_channel)))

    def get_gradient_magnitude(self, img):
        """"
        Compute the gradient magnitude for the passed in image
        :return image (numpy array)
        """
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(img, ddepth, 1, 0)
        dy = cv2.Sobel(img, ddepth, 0, 1)
        return np.sqrt(dx ** 2 + dy ** 2)

    def get_excess_green(self):
        """Compute the excess green image

        :return: 2-d array, ExG value for each pixel location
        """
        image_array = np.float32(self.bgr_modified) / 255
        # 2-d array, ExG value for each pixel location
        exg_green = image_array[:, :, 1] * 2 - image_array[:, :, 0] - image_array[:, :, 2]

        # normalize r, g, b values such that r + g + b = 1
        return scale_image(exg_green, -2, 2, -1, 1)

    def create_edge_mask(self, luminance_adjust, percent_strongest_edge=50):
        """
        create an edge mask from a bgr image with flattened luminance.  Uses the n % of the strongest edge
        pre: Has had the luminance flattened
        :param percent_strongest_edge, (0, 100], defaults to 50%
        :return: A mask created to encompass all of the sharpest edges and area close by
        """
        cpy_img = copy.deepcopy(self)
        cpy_img.set_luminance_lab(luminance_adjust)
        grad_img = self.get_gradient_magnitude(cpy_img.get_img_modified())

        # Scale the gradiant magnitude to avoid overflow
        #TODO change scale_image_auto to scale_image
        scaled_grad = scale_image_auto(grad_img, 0, 255, True, 'uint8')
        grad_grey = cv2.cvtColor(scaled_grad, cv2.COLOR_BGR2GRAY)

        num_strongest_edges = int(percent_strongest_edge/100 * len(grad_grey))
        # This will yield a fixed number of edges in relation to the size of the grayscale image
        # which is a function of the size of the initial *bgr* image input to the function
        indices = largest_indices(grad_grey, num_strongest_edges)

        mask = np.zeros(grad_grey.shape)
        mask[indices] = 255

        # The kernel size is also hardcoded, but doesn't ever need to be perfect, just pretty good
        kernel = np.ones((121, 121), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    def get_binary_image(self, median_threshold, median_luminance):
        """
        Creates a binary image to segment plants from non plant matter
        of the same dimensions as the Img object's image properties
        :pre: if the image needs flattened luminance, it's already done
        :param median_threshold: an int
        :return: An image (numpy array)
        """
        img_height, img_width, _ = self.get_img_modified().shape

        binary_image = np.zeros((img_height, img_width))
        scaled_exg = scale_image(self.get_excess_green(), -1, 1, 0, 255, rint=True, np_type='uint8')
        edge_mask = self.create_edge_mask(median_luminance)
        sample_img = scaled_exg[edge_mask == 255]

        thresh_val, _ = cv2.threshold(sample_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If Otsu finds a thresh too far from the median of the dataset, use the median.  This is needed for images where
        # there aren't any plants (single - mode histogram)
        if thresh_val + 1 < median_threshold:
            thresh_val = median_threshold
        # Threshold the image
        binary_image[scaled_exg > thresh_val] = 255
        binary_image = binary_image.astype(np.uint8)

        # connected components
        # find all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time
        # we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # 81 means we want blobs that are at least 9x9 squares (but don't have to be squares per se)
        min_size = 81
        thresh_img = np.zeros((output.shape))
        # for every component in the image, we keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                thresh_img[output == i + 1] = 255
        return thresh_img.astype('uint8')

    def add_row(self, number, line):

        self.Rows.append(Row(number, line))

    def straighten_image(self, rotation_angle):
        """ Note -- not currently in use, see straighten_rows"""
        straighten_angle = np.degrees(np.arctan(rotation_angle))
        image_center = tuple(np.array(self.get_img_initial().shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, straighten_angle, 1.0)
        result = cv2.warpAffine(self.get_img_initial(), rot_mat, self.get_img_initial().shape[1::-1], flags=cv2.INTER_LINEAR)
        self.bgr_img_initial = np.array(result)


class TranslatedImg(object):

    def __init__(self, img_file_path, homography_matrix, lines_in_image):
        self.img_file_path = img_file_path
        self.homography_matrix = homography_matrix
        self.list_of_lines = lines_in_image


class Row(object):
    """Row object that stores the initial cropping, regression fit, straightened row image, and the row number."""

    def __init__(self, number, initial_crop_coords):
        self.number = str(number)
        # initial_crop_coords should be a list of tuples, e.g. [(x1,y1),(x2,y2)]
        self.line = initial_crop_coords

    def get_row_slope(self):
        # equivalent to (y2 - y1) / (x2 - x1)
        return (self.line[1][1] - self.line[0][1]) / (self.line[1][0] - self.line[0][0])


class DataSet(object):

    def __init__(self, cam_height, cam_fov, cam_resolution, spacing, src, extension):
        self.cam_height = float(cam_height)
        self.cam_fov = float(cam_fov)
        self.cam_resolution = float(cam_resolution)
        self.plant_spacing_meters = float(spacing)
        self.img_paths = [os.path.join(src, name) for name in sorted(os.listdir(src))
                          if name.endswith(extension)]

        if len(self.img_paths) == 0:
            raise ValueError("Error: There are no image files in the src dir with the requested extension")

    def __find_static_objects(self):
        """
        precond: all images are the same dimensions
        :param list_img_paths: A list of paths to image files.
        :return: A grayscale img (32 bit float numpy array) with darkest regions where least change occurs
        """

        cumulative_img = np.float32(np.zeros(Img(self.img_paths[0]).bgr_img_initial[:, :, 0].shape))

        for i in range(0, len(self.img_paths)):
            cur_img = Img(self.img_paths[i - 1])
            prev_img = Img(self.img_paths[i])
            cumulative_img += np.sqrt(np.sum((np.float32(cur_img.bgr_img_initial)
                                              - np.float32(prev_img.bgr_img_initial)) ** 2, axis=2))
        return cumulative_img

    def create_mask_static_objects(self):
        """
        :return: A binary mask img (numpy array).
        """

        static_img = self.__find_static_objects()
        scaled_static_img = scale_image_auto(static_img, 0, 255, True, 'uint8')

        # create binary image of static objects
        binary_image = np.zeros(scaled_static_img.shape)
        # Could most likely apply Otsu here as the the histogram should be bimodal(?)
        binary_image[scaled_static_img > 25] = 0
        binary_image[scaled_static_img <= 25] = 255
        binary_image = binary_image.astype(np.uint8)

        # Dilate binary image to fill in holes
        kernel = np.ones((60, 60), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=1)

        # Values are swapped in order to use dilation, swap them back
        dilated[dilated < 1] = 1
        dilated[dilated > 1] = 0
        return dilated

    def get_median_luminance(self):
        """Return median luminance value for the data set"""
        avg_lum = 0
        for image_file in self.img_paths:
            avg_lum += Img(image_file).get_median_luminance_initial_img()
        return round(avg_lum/len(self.img_paths))

    def median_excess_green_threshold(self, luminance):
        """get the median threshold value found by using Otsu for the data set

        :return: int.
        """
        cumulative_thresh = []
        static_mask = self.create_mask_static_objects()
        # Compute the threshold value for EACH image in the dataset
        for img_path in self.img_paths:
            cur_img = Img(img_path)
            # Remove the static objects from the image
            cur_img.apply_mask(static_mask)
            # cur_img.set_luminance_lab(luminance)
            exg_cur_img = cur_img.get_excess_green()
            # Scaling the excess green image is needed for cv2's Otsu implementation
            scaled_exg = scale_image(exg_cur_img, -1, 1, 0, 255, rint=True, np_type='uint8')
            edge_mask_cur = cur_img.create_edge_mask(luminance)
            # Take a sampling of the strongest edge pixels.  This ensures the histogram is bimodal
            sample_img = scaled_exg[edge_mask_cur == 255]
            thresh_val, _ = cv2.threshold(sample_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Keep track of the cur value
            cumulative_thresh.append(thresh_val)

        # Return the most commonly occuring value in the array
        return np.median(np.array(cumulative_thresh))

    def get_median_thresh_value(self, luminance):
        """
        pre: The image is already had prec
        :return:
        """
        # This function could be changed in the future if exg isn't used exclusively
        return self.median_excess_green_threshold(luminance)

    def get_plant_spacing(self):
        return round(get_mean_px_per_meter_at_distance(self.cam_fov, self.cam_height, self.cam_resolution)
                     * self.plant_spacing_meters)

    def __homography_from_keypoint_matching(self, prev_im, new_im):
        """
        Match 2 images together via keypoint matching. Currently using SURF
        prev_im: IMG() object of previous image
        new_im: IMG() object of next/new image to overlay onto prev image
        returns: the homography matrix (M) that can be used to map from new_im to prev_im
        """
        # sift = cv2.xfeatures2d.SIFT_create()
        surf = cv2.xfeatures2d.SURF_create()

        # keypoint matching is done on grayscale images
        prev_gray = cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)

        # find keypoints in each image
        # maybe store keypoints and descriptors in img object so that prev image will already have them computed and then
        # they dont need to be recomputed here
        prev_kp, prev_desc = surf.detectAndCompute(prev_gray, None)
        # print('num key points:')
        # print(len(prev_kp), len(prev_desc))
        new_kp, new_desc = surf.detectAndCompute(new_gray, None)

        # flann is a faster/optimized way of finding matches, the brute-force way is to skip this and just use bf.knnMatch()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(new_desc, prev_desc, k=2)
        # matches = bf.knnMatch(new_desc, prev_desc, k=2)

        # ratio test -- this is recommended from the cv2 example I got most of this code from, there could be
        # potential for improvements here
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
            good.append(m)
        # print('num after ratio test:')
        # print(len(good))

        # extract matched keypoints
        src_pts = np.float32([new_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([prev_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M

    def get_homography(self, cur_img, prev_img):
        """Putting this in a wrapper to abstract how we get the homography matrix
        :param cur_img Img object with pre-processing applied
        :param prev_img, Img object with pre-processing applied
        :return homography matrix
        """
        return self.__homography_from_keypoint_matching(prev_img, cur_img)

    def transform_coords(self, x, transformation_matrix):
        """
        Let n be the number of coordinates.
        Let D be the dimensionality of the coordinates.

        :param x: n-by-D array of coords. D must be 2 or 3. x will be transformed to float64 before doing the transform.
        :param transformation_matrix: 3-by-3 transformation matrix, as returned by get_affine_transformation_matrix and
            get_homography_matrix.
        :return: n-by-D array of transformed coords corresponding to x.
        """
        x_shape = x.shape
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        x = np.float64(x)
        # cv2.perspectiveTransform requires the input matrix to be 3D for some reason.
        x_3d = x.reshape((1, *x.shape))
        x_prime_3d = cv2.perspectiveTransform(x_3d, transformation_matrix)
        x_prime = x_prime_3d.reshape(x_shape)
        return x_prime

    def get_klines_for_img(self, img_file_name, static_mask, median_luminance, med_thresh,
                           k=10, down_sample_factor=5, static_objects=False, verbose_plot=False):

        img = Img(img_file_name)

        if static_objects:
            img.apply_mask(static_mask)
        # Flatten the luminance
        img.set_luminance_lab(median_luminance)
        # Remove static objects
        # get_k_lines expects a binary image
        binary_img = img.get_binary_image(med_thresh, median_luminance)
        plant_spacing = self.get_plant_spacing()
        end_points = []

        # TODO: refactor this to have 750 replaced by automatically found plant spacing
        # TODO: refactor this to clean up the point coordinates
        for line in get_k_lines(binary_img, k, self.get_plant_spacing(), down_sample_factor, verbose_plot):
            intercept, slope = line
            x1 = 0
            x2 = binary_img.shape[1]
            y1 = binary_img.shape[0] - round(intercept)
            y2 = binary_img.shape[0] - round(slope *(x2-x1) + round(intercept))
            end_points.append((x1, y1))
            end_points.append((x2, y2))
        return np.array(end_points)


class CropRowsFromImg(object):

    def __init__(self, path_to_images, destination, camera_height, camera_fov,
                 camera_resolution, plant_spacing_meters, extension, has_static_objects):

        self.DataSet = DataSet(camera_height, camera_fov, camera_resolution,
                               plant_spacing_meters, path_to_images, extension)
        self.list_of_lines = []
        self.list_of_transformed_images = []
        self.static_mask = self.DataSet.create_mask_static_objects() if has_static_objects else None
        self.med_lum  =    self.DataSet.get_median_luminance()
        self.med_thresh =  self.DataSet.get_median_thresh_value(self.med_lum)
        self.plant_spacing = self.DataSet.get_plant_spacing()
        self.src = path_to_images
        self.destination = destination

    def turn_array_coords_into_list_of_lines(self, np_coords):
        """Essentially turns a (n,2) numpy array into a list of line segment end points
            e.g. a list of values of the form (x1, y1), (x2, y2)"""
        list_of_lines = []
        np_coords = np_coords.tolist()
        for i in range(0, len(np_coords)-1, 2):
            list_of_lines.append((np_coords[i], np_coords[i+1]))
        return list_of_lines

    def compare_closeness_line_segments(self, line_1, line_2):
        """This method is based on the assumption that the closest two non-parallel line segments that don't interssect
        can be is either of the endpoints. e.g. it returns the smaller of (line_1's y1 - line_2's y2 and
        line_1's y2 - line_2's y2)

        :param line_1 a line in the form [(x1, y1), (x2, y2)]
        :param line_2 a line in the form [(x1, y1), (x2, y2)]"""

        # This returns the smallest distance of two endpoints in terms Y-value (X-val is ignored as they are the same)
        # return min(abs(line_1[0][1]-line_2[0][1]), abs(line_1[1][1]-line_2[1][1]))
        return abs(line_1[0][1] - line_2[0][1])/2 + abs(line_1[1][1] - line_2[1][1])/2

    def is_new_line(self, line, list_of_lines):
        """Checks if the passed in line is a new line based off how far away it is from established lines"""
        print("Checking line:", line)
        for l in list_of_lines:
            # 0.80 is a magic number, but it's something we justify by defining it as a buffer for error in registration
            print("    ", l, self.compare_closeness_line_segments(line, l), )
            if self.compare_closeness_line_segments(line, l) <   self.plant_spacing * 0.8:
                print("    False")
                return False
        print("    True")
        return True

    def find_initial_lines(self, logging=False):
        """Currently written to start on the second image such that the homography matrix is done pairwise from the
        second image on as cur / previous comparison."""

        # Ensure that the list of found lines / translated images are empty
        self.list_of_lines = []
        self.list_of_transformed_images = []

        cur_img = Img(self.DataSet.img_paths[0])
        cur_img_k_lines = self.DataSet.get_klines_for_img(cur_img.path, self.static_mask, self.med_lum, self.med_thresh)
        cur_img_k_lines_as_list = self.turn_array_coords_into_list_of_lines(cur_img_k_lines)
        # We want to store a list of translated objects so we don't have to calculate the homography matrix again
        # When we want to assign rows to images
        homography_prev_to_cur = self.DataSet.get_homography(cur_img.get_img_initial(), cur_img.get_img_initial())
        self.list_of_transformed_images.append(TranslatedImg(cur_img.path, homography_prev_to_cur, cur_img_k_lines_as_list))
        # We assume the first images rows have not been seen before, and thusly can be all added to our "global" list
        row_image_pairs = [(l, [cur_img.path]) for l in cur_img_k_lines_as_list]
        self.list_of_lines.extend(row_image_pairs[::-1])
        print("Lines from first image", cur_img_k_lines_as_list[::-1])

        for image_file_path in self.DataSet.img_paths[1:]:

            prev_img = cur_img
            prev_img_lines = cur_img_k_lines_as_list
            cur_img = Img(image_file_path)

            homography_prev_to_cur = self.DataSet.get_homography(cur_img.get_img_initial(), prev_img.get_img_initial())

            cur_img_k_lines = self.DataSet.get_klines_for_img(cur_img.path, self.static_mask, self.med_lum, self.med_thresh)
            cur_img_k_lines_as_list = self.turn_array_coords_into_list_of_lines(cur_img_k_lines)

            # If we found a row
            if len(cur_img_k_lines_as_list) > 0:
                # We perform the transform pairwise to see if any rows in the cur_img are not in the prev_img
                cur_img_updated_pairwise = self.DataSet.transform_coords(cur_img_k_lines, homography_prev_to_cur)
                cur_img_lines_pairwise = self.turn_array_coords_into_list_of_lines(cur_img_updated_pairwise)

                # We might not need to store the potential lines here.  It might be correct to skip it and find them
                # later using a homography matrix to map back the result to that image
                self.list_of_transformed_images.append(TranslatedImg(cur_img.path, homography_prev_to_cur,cur_img_k_lines_as_list))
                print(cur_img.filename, "k-lines found", cur_img_k_lines_as_list)

                # Add all the new lines in the current image to the global list, usually 0 or 1
                for i in range(len(cur_img_lines_pairwise)-1, -1, -1):
                    if self.is_new_line(cur_img_lines_pairwise[i], prev_img_lines):
                       self.list_of_lines.append((cur_img_k_lines_as_list[i],[]))


                # self.list_of_lines contains all the lines in the cur img, now just need to update filepaths for the rows
                # found in cur_img.
                for i in range(0, len(cur_img_lines_pairwise)):
                    self.list_of_lines[-1 - i][1].append(cur_img.path)

                print("CURRENT NUMBER OF DETECTED ROWS", len(self.list_of_lines))

            if logging:
                # Dump the list of lines per iteration of the algorithm
                pickle.dump(self.list_of_lines, open("/home/josh/pickeld_objects/list_of_lines/rows.p", "wb"))
                # Dump the current image's path and homography matrix that relates it to the first image.
                pickle.dump(((cur_img.path, homography_prev_to_cur, homography_prev_to_cur)),
                            open("/home/josh/pickeld_objects/list_of_transformed_images/homographys.p", "wb"))

    def straighten_rows(self, img_arr, lines, height):
        """Straightens an image based on the given regression/centering line to the desired height."""
        # img_arr = Img.get_img_initial()

        # for RGB / BGR
        if len(img_arr.shape) == 3:
            img_height, img_width, _ = img_arr.shape
        # for binary / greyscale
        elif len(img_arr.shape) == 2:
            img_height, img_width = img_arr.shape
        else:
            raise ValueError("Tried to get the shape of an image array that isn't 3 or 1 dimensions")

        rows = []
        for line in lines:
            # line is [(x1,y1), (x2,y2)]
            y1 = line[0][1]
            y2 = line[1][1]
            src_points = np.float32([[0, y1 - int(height / 2)], [img_width, y2 - int(height / 2)],
                                     [img_width, y2 + int(height / 2)], [0, y1 + int(height / 2)]])
            dst_points = np.float32([[0, 0], [img_width, 0],
                                     [img_width, height], [0, height]])

            img_array = np.array(img_arr)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            dst = cv2.warpPerspective(img_array, M, (img_width, height))
            rows.append(np.array(dst))

        return rows

    def is_row_in_img(self, row, path_to_image, homography_matrix):
        """
        *This is now depricated, we keep track of which row is in which image in the find_initial_lines() method*
        :param row: a line in the form of [(x1, y1), (x2,y2)]
        :param path_to_image: Must be a full path
        :param homography_matrix: The homography matrix that maps the image back to the start of the globlal coordinate
                sytem
        :return: True if the row is in the image
        """

        img = Img(path_to_image)

        img_height, img_width, _ = img.get_img_initial().shape

        initial_image_coordinates = np.array([(0, 0), (img_width, 0), (0, img_height), (img_width, img_height)])
        updated_image_coordinates = self.DataSet.transform_coords(initial_image_coordinates, homography_matrix)

        y_dimensions = [y[1] for y in updated_image_coordinates]
        x_dimensions = [x[0] for x in updated_image_coordinates]

        img_as_polygon = box(min(x_dimensions), min(y_dimensions), max(x_dimensions), max(y_dimensions))

        min_row_y = min(row[0][1], row[1][1]) - 25
        max_row_y = max(row[0][1], row[1][1]) + 25

        min_point = Point(statistics.median(x_dimensions), min_row_y)
        max_point = Point(statistics.median(x_dimensions), max_row_y)
        return img_as_polygon.contains(min_point) and img_as_polygon.contains(max_point)

    def crop_rows_in_img(self):
        for t_im in self.list_of_transformed_images:
            cur_img = Img(t_im.img_file_path)
            row_num = 1
            local_row_counter = 1
            for i in range(0, len(self.list_of_lines)):
                if cur_img.path in self.list_of_lines[i][1]:
                    cur_line = t_im.list_of_lines[len(t_im.list_of_lines) - local_row_counter]
                    line_as_arry = np.array(cur_line)
                    transformed_line = line_as_arry.tolist()
                    cur_img.add_row(str(row_num), transformed_line)
                    local_row_counter += 1
                row_num += 1
            row_lines = [r.line for r in cur_img.Rows]
            cropped_cur_color = self.straighten_rows(cur_img.get_img_initial(), row_lines, 500)
            # Need to create the binary used to find the rows via k-lines
            # cur_img.apply_mask(self.static_mask)
            cropped_binary = self.straighten_rows(cur_img.get_binary_image(self.med_thresh, self.med_lum), row_lines, 500)
            row_nums = [row.number for row in cur_img.Rows]
            row_vals_color = list(zip(row_nums, cropped_cur_color))
            row_vals_binary = list(zip(row_nums, cropped_binary))
            # Refactor to use a zipped list for row_vals_color / row_vals_binary
            for num, row_img in row_vals_color:
                plt.imsave(self.destination+num+"_"+cur_img.filename, row_img[:, :, (2, 1, 0)])
            for num, row_img in row_vals_binary:
                cv2.imwrite(os.path.join(self.destination, num+"_"+cur_img.filename+"_binary_.png"), row_img)

    def print_transformed_img_dimensions(self):
        """This is now depricated, it was used to inspect the dimensions of each image in the global stitch"""
        img_height, img_width = 3000, 4000
        initial_image_coordinates = np.array([(0, 0), (img_width, 0), (0, img_height), (img_width, img_height)])
        for t in self.list_of_transformed_images:
            path = t.img_file_path
            print(path)
            print(self.DataSet.transform_coords(initial_image_coordinates, t.homography_matrix))

    def stich_data_set(self, end_offset=None, apply_lines=False):
        """
        Creates a stich of the data set
        :param end_offset: The last image in the dataset
        :param apply_lines: Set to true if you want the k-lines applied to the stitch
        :return: np array (image)
        """

        prev_img = Img(self.DataSet.img_paths[61])
        cur_img  = Img(self.DataSet.img_paths[62])

        initial_width = prev_img.get_img_initial().shape[0]
        initial_height = prev_img.get_img_initial().shape[1]

        # [0] for x values, [1] for y values
        dimensions = [[0 , initial_width], [initial_height , 0]]

        image_vals = []

        image_vals.append([prev_img.path, None])

        homography_matrix = self.DataSet.get_homography(cur_img, prev_img)
        cur_height, cur_width, _ = cur_img.get_img_initial().shape
        # Check each corner of the image
        cur_img_dimensions = np.array([[0,0], [0, cur_width], [cur_height, 0], [cur_height, cur_width]])
        # Warp the cur image to see if it extentds past the current maximums / minimums
        cur_img_dimensions = self.DataSet.transform_coords(cur_img_dimensions, homography_matrix)

        for point in cur_img_dimensions:
            dimensions[0].append(point[0])
            dimensions[1].append(point[1])

        image_vals.append([cur_img.path, homography_matrix])

        stopping_offset = len(self.DataSet.img_paths)-1 if end_offset is None else end_offset

        for img in self.DataSet.img_paths[62:stopping_offset]:
            prev_img = cur_img
            cur_img = Img(img)
            # From original image to cur
            homography_matrix =  np.matmul(homography_matrix, self.DataSet.get_homography(cur_img, prev_img))
            cur_height, cur_width, _ = cur_img.get_img_initial().shape
            # Check each corner of the image
            cur_img_dimensions = np.array([[0, 0], [0, cur_width], [cur_height, 0], [cur_height, cur_width]])
            cur_img_dimensions = self.DataSet.transform_coords(cur_img_dimensions, homography_matrix)
            for point in cur_img_dimensions:
                dimensions[0].append(point[0])
                dimensions[1].append(point[1])
            # I like this print statement, but it could be replaced with a tqdm for-loop for a sense of progress
            print(cur_img.filename, min(dimensions[0], key=float), max(dimensions[0], key=float),
                                                        min(dimensions[1], key=float), max(dimensions[1], key=float))
            image_vals.append([cur_img.path, homography_matrix])

        # The dimensions of the overall stitch will be the maximum x,y  and minimum x,y values f
        # found by applyng the registration back to the first image in the data set.
        img_width = abs(int(round(max(dimensions[0], key=float)))) +  abs(int(round(min(dimensions[0], key=float))))
        img_height = abs(int(round(max(dimensions[1], key=float)))) + abs(int(round(min(dimensions[1], key=float))))

        total_img = np.zeros((img_height, img_width, 3), dtype=cur_img.get_img_modified().dtype)

        # top left, bottom left, bottom right, top right
        first_img_dimensions = np.array([(0,0), (initial_width,0), (initial_width,initial_height), (0,initial_height)])

        # An array with the original images dimensions updated to where the are in the stitched image.
        # Values in the array are tuples representing the top left, top right, bottom right, bottom left.
        final_img_dimensions = np.array([(0-abs(int(min(dimensions[0], key=float))), 0),
                                        (initial_height-abs(int(min(dimensions[0], key=float))), 0),
                                         (initial_width-abs(int(min(dimensions[1], key=float))), initial_height),
                                         (0-abs(int(min(dimensions[0], key=float))), initial_height)])

        # This homography matrix updates the initial images coordinates to the stitched ortho
        h0_to_g = get_homography_matrix(first_img_dimensions, final_img_dimensions)
        # Update every homography matrix to originate from the stitched image
        for m in image_vals[1:]:
            # ordering here might be wrong
            m[1] = np.matmul(h0_to_g, m[1])

        k_lines = []

        for im_val in image_vals[1:]:
            cur_img = Img(im_val[0])
            k_lines_cur_img = self.DataSet.get_klines_for_img(im_val[0], self.static_mask, self.med_lum, self.med_thresh)
            k_lines.append((k_lines_cur_img, im_val[1]))
            # Warp the cur image's dimensions to match overall image.
            warped_cur = warp_image(cur_img.get_img_modified(), total_img.shape[0:2], im_val[1])
            new_pixels = warped_cur != 0
            total_img[new_pixels] = warped_cur[new_pixels]

        if apply_lines:
            for line in k_lines:
                #update the lines to the warped image
                transformed_line = self.DataSet.transform_coords(line[0], line[1])
                list_of_lines = self.turn_array_coords_into_list_of_lines(transformed_line)
            for l in list_of_lines:
                # Drawing purple, I chose this for the contrast with green and it's the same in BGR as RGB
                total_img = cv2.line(total_img, (int(l[0][0]), (int(l[0][1]))),
                                   (int(l[1][0]), (int(l[1][1]))), (255, 0, 255), 25)

        return total_img

    def stitch_base_case(self):
        """
        pre: all images in the data set are of the same dimensions
        :return: A global ortho done pairwise.
        """
        # num_to_stitch = len(self.DataSet)
        # while(num_to_stitch != 1 ):
        stitched_images = []
        num_total = len(self.DataSet)
        for i in range(0, num_total, 2):
        # for i in range(0, len(self.DataSet.img_paths)+1, 2):
            prev = Img(self.DataSet.img_paths[i])
            cur = Img(self.DataSet.img_paths[i+1])
            initial_height, initial_width, _ = prev.get_img_initial().shape
            homography_matrix = self.DataSet.get_homography(cur.get_img_initial(), prev.get_img_initial())
            h_to_self = self.DataSet.get_homography(prev.get_img_initial(), prev.get_img_initial())

            # Top left, top right, bottom right, bottom left
            cur_img_dimensions = np.array([(0,0), (initial_width,0), (initial_width,initial_height), (0,initial_height)])
            cur_img_dimensions = self.DataSet.transform_coords(cur_img_dimensions, homography_matrix)
            # We need all the dimensons of the cur image transformed in case the image orientation is any any possible direction
            max_x_cur = np.max([x[0] for x in cur_img_dimensions])
            min_x_cur = np.min([x[0] for x in cur_img_dimensions])
            max_y_cur = np.max([y[1] for y in cur_img_dimensions])
            min_y_cur = np.min([y[1] for y in cur_img_dimensions])
            stitch_width = abs(int(round(max(max_x_cur, initial_width)))) + abs(int(round(min(min_x_cur, 0))))
            stitch_height = abs(int(round(max(max_y_cur, initial_height)))) + abs(int(round(min(min_y_cur, 0 ))))
            total_img = np.zeros((stitch_height, stitch_width, 3), dtype=cur.get_img_initial().dtype)


            warped_prev = warp_image(prev.get_img_initial(), total_img.shape[0:2], h_to_self)
            prev_pixels = warped_prev != 0
            total_img[prev_pixels] = warped_prev[prev_pixels]

            warped_cur = warp_image(cur.get_img_modified(), total_img.shape[0:2], homography_matrix)
            new_pixels = warped_cur !=0
            total_img[new_pixels] = warped_cur[new_pixels]

            stitched_images.append(total_img)
            print(len(stitched_images)*2, "/", len(self.DataSet.img_paths)//2)

        return stitched_images

    def stitch_normal_case(self, list_of_imgs):
        """Given a list of images, stitch ever other image together"""
        img_height, img_width, _ = list_of_imgs[0].shape
        stitched_images = []

        for i in range(0, len(list_of_imgs), 2):
            prev = list_of_imgs[i]
            cur = list_of_imgs[i+1]
            h_to_self = self.DataSet.get_homography(prev, prev)
            homography_matrix = self.DataSet.get_homography(cur, prev)
            cur_img_dimensions = np.array(
                [(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)])
            cur_img_dimensions = self.DataSet.transform_coords(cur_img_dimensions, homography_matrix)
            # We need all the dimensons of the cur image transformed in case the image orientation is any any possible direction
            max_x_cur = np.max([x[0] for x in cur_img_dimensions])
            min_x_cur = np.min([x[0] for x in cur_img_dimensions])
            max_y_cur = np.max([y[1] for y in cur_img_dimensions])
            min_y_cur = np.min([y[1] for y in cur_img_dimensions])
            stitch_width = abs(int(round(max(max_x_cur, img_width)))) + abs(int(round(min(min_x_cur, 0))))
            stitch_height = abs(int(round(max(max_y_cur, img_height)))) + abs(int(round(min(min_y_cur, 0))))
            total_img = np.zeros((stitch_height, stitch_width, 3), dtype=cur.dtype)

            warped_prev = warp_image(prev, total_img.shape[0:2], h_to_self)
            prev_pixels = warped_prev != 0
            total_img[prev_pixels] = warped_prev[prev_pixels]

            warped_cur = warp_image(prev, total_img.shape[0:2], homography_matrix)
            new_pixels = warped_cur != 0
            total_img[new_pixels] = warped_cur[new_pixels]
            stitched_images.append(total_img)

        return stitched_images

    def create_binary_stitch(self):


        stitched = self.stitch_base_case()
        stitched_imgs = []
        to_stitch = len(self.DataSet.img_paths)
        # to_stitch = len(self.DataSet.img_paths)
        # This loop should be done log(n) times.  Each iteration reduces to_stitch by half
        while to_stitch != 1:
            print(len(stitched_imgs))
            stitched_imgs = self.stitch_normal_case(stitched)
            to_stitch = len(stitched_imgs)

        return stitched_imgs

    def write_out_params(self):
        """Writes a json file of the important parameters to a json file at the destination dir chosen at the objects
            instantiation.
        """
        # TODO add field information (plant spacing in particular)
        param_dict = {
            "Source Directory": self.src,
            "Output Directory": self.destination,
            "Plant Spacing": self.plant_spacing,
            "Camera Height": self.DataSet.cam_height,
            "Camera Field of View": self.DataSet.cam_fov,
            "Vertical Resolution": self.DataSet.cam_resolution,
            # TODO derive this param
            "K value": 10,
            "Median Luminance Value": self.med_lum,
            "Median Threshold Value": self.med_thresh,
            "Number of Rows Detected": len(self.list_of_lines),
            "Rows in Images": self.list_of_lines
        }

        with open(os.path.join(self.destination, "params.json"), 'w') as fp:
            json.dump(obj=param_dict, fp=fp, indent=4)

    def run_rowcropper(self):

        self.find_initial_lines()
        self.crop_rows_in_img()
        self.write_out_params()


def main(src, destination, cam_height, cam_fov, cam_resolution, plant_spacing_m, extension, has_static_objects):

    row_cropper = CropRowsFromImg(src, destination, cam_height, cam_fov,
                                  cam_resolution, plant_spacing_m, extension, has_static_objects)
    row_cropper.run_rowcropper()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A tool to crop rows of plants from raw images")
    parser.add_argument('input', type=str, help="A json file containing all of the parameters needed to run the tool")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        inputs = json.load(f)

    if not os.path.isdir(inputs["input_directory"]):
        raise ValueError("Error:  Input directory does not exist")
    elif not os.path.isdir(inputs["desination_directory"]):
        raise ValueError("Error:  Destination directory doesn't exist")

    main(
        inputs["input_directory"],
        inputs["desination_directory"],
        inputs["camera_height"],
        inputs["camera_field_of_view"],
        inputs["cam_vertical_resolution"],
        inputs["row_spacing"],
        inputs["img_extension"],
        inputs["contains_static_objects"]
    )