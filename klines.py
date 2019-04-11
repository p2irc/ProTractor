import time
import cv2
import numpy as np
import statsmodels.api as sm

from matplotlib import pyplot as plt


def distance_from_line(point, line):
    """
    Computes the distance along the y-axis (vertical distance) of a point from a line
    point: (x, y)
    line: Line() object
    """
    return (point[1] - (line.intercept + point[0]*line.slope))**2


def std_line(p1, p2):
    """
    Converts a line into standard form (with a -C value)
    A line here is represented by two points that it passes through
    p1: (x, y) point
    p2: (x, y) point
    """
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0]*p2[1] - p2[0]*p1[1])
    return a, b, -c


def intersection(l1, l2):
    """
    Finds the intersection of two lines
    The std_line() function returns the form that L1 and L2 are expected to be in
    L1: standard form of a line as (A, B, -C)
    L2: standard form of a line as (A, B, -C)
    """
    d = l1[0] * l2[1] - l1[1] * l2[0]
    dx = l1[2] * l2[1] - l1[1] * l2[2]
    dy = l1[0] * l2[2] - l1[2] * l2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return x, y
    else:
        return False


def shortest_distance_two_lines(line_1, line_2, width):
    """
    Computes the shortest distance between two line segments
    line_1: Line() object
    line_2: Line() object
    """
    # essentially check how close the endpoints are of line segments AB and CD
    # we check elsewhere for intersecting lines

    # distance of endpoint A from line segment CD
    dist_1 = np.abs(line_1.intercept - line_2.intercept)
    # distance of endpoint B from line segment CD
    dist_2 = np.abs((line_1.intercept + line_1.slope*width) - (line_2.intercept + line_2.slope*width))
    return np.min([dist_1, dist_2])


class Line(object):
    """
    The 'center' of a cluster
    This object tracks the intercept, slope, width, standard form, and nearest points
    width is the width of the plane/frame/image
    color is for plotting/testing
    """
    def __init__(self, intercept, slope, width, color):
        self.intercept = intercept
        self.slope = slope
        self.std_form = std_line([0, intercept], [width, intercept+width*slope])
        self.points = []
        self.color = color


class KLines(object):
    """
    The overall model with all needed methods
    fit_k_lines_to_data() will perform the full fitting and report final lines
    """
    def __init__(self, data, k, height, width):
        """
        data: rows of (x, y) pairs (2d)
        """
        self.lines = None
        self.clusters = np.zeros(len(data))
        self.data = data
        self.delta = 10  # just a non-zero value
        self.k = k
        self.height = height
        self.width = width

    def initialize_lines(self):
        """
        Currently preferring the horizontal grid initialization
        If using plotting, there are only 7 colors allowed atm
        If wanting to use a k>7, need to change line_colors variable
        """
        line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'coral']

        # horizontal grid
        gap = self.height // self.k
        intercepts = [i*gap for i in range(1, self.k+1)]
        slopes = [0] * self.k

        self.lines = []
        for intercept, slope, color in zip(intercepts, slopes, line_colors):
            self.lines.append(Line(intercept, slope, self.width, color))

    def assign_points_to_clusters(self):
        """
        Assigns the points in self.data to clusters based on distance from lines
        """
        # reset points 'belonging' to lines so they aren't double counted
        for line in self.lines:
            line.points = []

        for point in self.data:
            # get distance from point to each line
            distances = []
            for line in self.lines:
                dist = distance_from_line(point, line)
                distances.append(dist)
            # smallest distance is the line/cluster we assign the point to
            cluster = np.argmin(np.array(distances))
            self.lines[cluster].points.append(point)

    def del_line_from_close_lines(self, spacing):
        """
        If two lines are close (within the row pixel distance) then we delete the line with fewer points
        The assign_points_to_clusters() method should be called after this one (maybe add that to the function?)
        to ensure that the points from deleted lines will get reassigned immediately
        """
        lines_to_delete = [] # don't want to delete from something you are iterating over
        for i in range(len(self.lines)):
            for j in range(i+1, len(self.lines)):
                # find the shortest distance between the 2 lines
                distance = shortest_distance_two_lines(self.lines[i], self.lines[j], self.width)
                if distance < spacing: # PARAM
                    # delete line based on which has more points assigned to it
                    if len(self.lines[i].points) > len(self.lines[j].points):
                        # delete line j since it has fewer points
                        if j not in lines_to_delete:
                            lines_to_delete.append(j)
                    else:
                        # delete line i since it has fewer points
                        if i not in lines_to_delete:
                            lines_to_delete.append(i)
        # now we actually delete, doing it in reverse order precents issues with deleting
        # from something as we are iterating over it
        for i in sorted(lines_to_delete, reverse=True):
            del self.lines[i]

    def del_line_from_intersecting_lines(self):
        """
        If two lines are intersecting within the image then we delete the line with fewer points
        The assign_points_to_clusters() method should be called after this one (maybe add that to the function?)
        To ensure that the points from deleted lines will get reassigned immediately
        """
        lines_to_delete = []  # don't want to delete from something you are iterating over
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                # find the intersection
                R = intersection(self.lines[i].std_form, self.lines[j].std_form)
                if R and R[0] > 0 and R[0] < self.width:  # if there is an intersection within the frame of view PARAM
                    # delete line based on which has more points assigned to it
                    if len(self.lines[i].points) > len(self.lines[j].points):
                        # delete line j since it has fewer points
                        if j not in lines_to_delete:
                            lines_to_delete.append(j)
                    else:
                        # delete line i since it has fewer points
                        if i not in lines_to_delete:
                            lines_to_delete.append(i)
        # now we actually delete, doing it in reverse order prevents issues with deleting
        # from something as we are iterating over it
        for i in sorted(lines_to_delete, reverse=True):
            del self.lines[i]

    def reassign_points_from_intersecting_lines(self):
        """
        If two lines are intersecting within the image then we take the points from the line with
        fewer points and give them to the line with more points
        The remove_lines_with_no_points() method should be called after this one (maybe add that to the function?)
        """
        for i in range(len(self.lines)):
            for j in range(i+1, len(self.lines)):
                # find the intersection
                r = intersection(self.lines[i].std_form, self.lines[j].std_form)
                # if there is an intersection within the frame of PARAM
                if r and self.width < r[0] > 0:
                    # reassign cluster values to the cluster/line with more points
                    if len(self.lines[i].points) > len(self.lines[j].points):
                        # line i has more points, so give it line j's points
                        self.lines[i].points.extend(self.lines[j].points)
                        self.lines[j].points = []
                    else:
                        # line j has more points, so give it line i's points
                        self.lines[j].points.extend(self.lines[i].points)
                        self.lines[i].points = []

    def reassign_points_from_close_lines(self, min_line_distance):
        """
        If two lines are close (within the row pixel distance) then we take the points from the line with
        fewer points and give them to the line with more points
        The remove_lines_with_no_points() method should be called after this one (maybe add that to the function?)
        """
        for i in range(len(self.lines)):
            for j in range(i+1, len(self.lines)):
                # find the shortest distance between the 2 lines
                distance = shortest_distance_two_lines(self.lines[i], self.lines[j], self.width)
                if distance < min_line_distance:
                    # reassign cluster values to the cluster/line with more points
                    if len(self.lines[i].points) > len(self.lines[j].points):
                        # line i has more points, so give it line j's points
                        self.lines[i].points.extend(self.lines[j].points)
                        self.lines[j].points = []
                    else:
                        # line j has more points, so give it line i's points
                        self.lines[j].points.extend(self.lines[i].points)
                        self.lines[i].points = []

    def remove_lines_with_no_points(self):
        """
        If a line has 1 or fewer points we remove it since it won't be able to
        fit a unique line anymore
        """
        new_lines = []
        for line in self.lines:
            if len(line.points) > 0:
                new_lines.append(line)
        self.lines = new_lines

    def fit_new_lines(self):
        """
        Refits each line to its points
        """
        # probably need a more efficient way of making points fit into the form
        # that sm wants for fitting lines. Maybe need to store points in this form
        # so we don't have to loop here
        for line in self.lines:
            xs = []
            ys = []
            for point in line.points:
                xs.append(point[0])
                ys.append(point[1])
            if len(xs) > 1:
                x = sm.add_constant(xs)  # so it fits a regression line with an intercept
                model = sm.OLS(ys, x)
                results = model.fit()
                # This is problematic and should be adressed later.
                if len(results.params) == 2:
                    b0, b1 = results.params
                    line.intercept = b0
                    line.slope = b1
                    line.std_form = std_line([0, line.intercept], [self.width, line.intercept + self.width * line.slope])

    def update_delta(self, old_lines):
        """
        Computes the change from old_lines to the current lines
        old_lines needs to be stored before lines are refit (see how its done in fit_k_lines_to_data())
        old_lines: list of Line() objects
        """
        update_delta = 0
        for i in range(len(self.lines)):
            update_delta += (self.lines[i].intercept - old_lines[i][0])**2 + (self.lines[i].slope - old_lines[i][1])**2
        self.delta = update_delta

    def plot_lines(self, offset):
        """
        Was used to plot lines and their points for testing
        """
        for line in self.lines:
            xs = [point[0] for point in line.points]
            ys = [point[1] for point in line.points]
            plt.scatter(xs, ys, color=line.color)
            x = np.linspace(0, self.width, self.width*1.5)
            plt.plot(x, line.intercept + line.slope * x, color=line.color)

        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)
        plt.show()
        plt.close()


def fit_k_lines_to_data(data, k, min_line_distance, size, verbose_plot=False):
    """
    Performs the fitting of lines to the given data
    data: (x,y) pairs (2d)
    k: how many lines to start with
    size: [height, width] of the plane (or the image)
    verbose_plot: boolean, indicates whether to plot each step (used for testing)
    returns: a list of (intercept, slope) pairs of the final fitted lines
    """
    # initialize model
    model = KLines(data, k, size[0], size[1])
    model.initialize_lines()
    model.assign_points_to_clusters()
    # verbose_plot will plot the stages for visual inspection
    if verbose_plot:
        # plot initial lines
        line_color = ['r', 'g', 'b', 'c', 'm', 'y', 'coral']
        plt.close()
        plt.scatter(data[:, 0], data[:, 1], color='k')
        for i in range(len(model.lines)):
            x = np.linspace(0, size[1], size[1]*1.5)
            plt.plot(x, model.lines[i].intercept + model.lines[i].slope * x, color=line_color[i])
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)
        plt.show()

    i = 1
    # burn-in
    # before we start checking to eliminate intersecting and/or close lines
    for q in range(5):  # PARAM - will be tuned based on speed of program'
        # assign each point to the nearest line
        if verbose_plot:  # plot assignments
            model.plot_lines(i)
            i+=1
        # remove lines that have no points
        model.remove_lines_with_no_points()
        # fit new lines
        model.fit_new_lines()
        if verbose_plot:  # plot update
            model.plot_lines(i)
            i+=1
    # iterate until convergence
    model.delta = 10  # needs to be non-zero just to get started
    while model.delta != 0:  # we stop iterating when lines don't update
        # store old lines -- used to test for convergence
        old_lines = [(line.intercept, line.slope) for line in model.lines]

        # assign each point to the nearest line
        # (since previous loop will have fit new lines at the end)
        model.assign_points_to_clusters()
        model.remove_lines_with_no_points()
        if verbose_plot:
            model.plot_lines(i)
            i+=1

        # remove intersecting lines
        # model.reassign_points_from_intersecting_lines()
        model.del_line_from_intersecting_lines()
        model.assign_points_to_clusters()
        if verbose_plot:
            model.plot_lines(i)
       #  model.remove_lines_with_no_points()
        model.assign_points_to_clusters()
        if verbose_plot:
            model.plot_lines(i)
            i+=1
        model.fit_new_lines()
        if verbose_plot:
            model.plot_lines(i)
            i+=1

        # remove close lines
        # model.reassign_points_from_close_lines(min_line_distance)
        model.del_line_from_close_lines(min_line_distance)
        model.assign_points_to_clusters()
        if verbose_plot:
            model.plot_lines(i)
            i += 1
        # model.remove_lines_with_no_points()
        # model.assign_points_to_clusters()
        if verbose_plot:
            model.plot_lines(i)
            i += 1
        if verbose_plot:
            model.plot_lines(i)
            i += 1
        # compute change in lines to determine when to stop iterating (convergence)
        model.update_delta(old_lines)

    # we return lines as (intercept, slope) pairs
    return_lines = []
    for line in model.lines:
        return_lines.append([line.intercept, line.slope])

    return return_lines


def __largest_indices(arr, n):
    """Return the n largest indices of an array"""

    # Flattens nd-arrays, does nothing on a 1-d array
    flattened_array = arr.flatten()
    # Partition the array into the indices of largest values
    indices = np.argpartition(flattened_array, -n)[-n:]
    # Sorts in ascending order
    indices = indices[np.argsort(-flattened_array[indices])]
    return np.unravel_index(indices, arr.shape)


def get_k_lines(img, k, min_line_distance, down_sample_factor, verbose_plot=False):
    """
    pre: down_sample_factor is an integer and greater or equal to 1.

    :param img: A Binary img
    :param k: The number of seeded lines (int)
    :param min_line_distance: int
    :param down_sample_factor: A non-negative integer factor to downsample the image by.
            can be seen as a quality dial
    :param verbose_plot: Plots the final result and intermediate results of the algorithm
    :return: A list of lists where each sub list contains the Y-intercept and slope of the found line
    """

    # Enforce the preconditions
    if down_sample_factor < 0 or isinstance(down_sample_factor, float):
        raise ValueError("Can't down sample by a negative number or floating point number")

    # TODO -- Generalize the 0.01 value.  Essentialyl we just want there to be a minium number of plant matter pixels
    # in an image to bother running k-lines, this value should also be parameterized
    if cv2.countNonZero(img)  < 0.0009 * img.shape[0] * img.shape[1]:
        return []

    min_line_distance = min_line_distance // 2
    # Need to flip the image by 180 degrees to get lines with proper sloper intercept
    # Could refactor this later, but for now i'll just leave it as a flip
    img = np.flipud(img)
    img_grey = img
    # down sample the image by the scaling factor param
    img_grey_height, img_grey_width = img_grey.shape
    down_sample_height = img_grey_height // down_sample_factor
    down_sample_width = img_grey_width // down_sample_factor
    img_grey = cv2.resize(img_grey, (down_sample_width, down_sample_height))
    # Minimum distance between rows will need to be scaled to match down sampled image
    down_sample_min_distance = min_line_distance // down_sample_factor

    non_zero_pixels = cv2.countNonZero(img_grey)
    non_zero_indices = __largest_indices(img_grey, non_zero_pixels)

    # Create a list of x,y pairs
    data_points = np.array(list(zip(non_zero_indices[1], non_zero_indices[0])))

    k_line_values = fit_k_lines_to_data(data_points, k, down_sample_min_distance,
                                      [down_sample_height, down_sample_width], verbose_plot)

    # Ensure that the downsample intercepts are scaled to match the dimensions of the input image
    for line in k_line_values:
        line[0] *= down_sample_factor

    return k_line_values
