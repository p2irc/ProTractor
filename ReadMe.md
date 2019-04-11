## Requirements ##
  * Python 3.*
  * Opencv with SURF enabled
  * Numpy
  * Requirements that can be installed with pip are in requirements.txt

## Using the system ##

Prior to using the row cropper, you must first create an input JSON file containing all of the
required field parameters.  These parameters are, a path to the directory containing the images,
a path to the directory where the cropped images will be written, the space in meters between the rows,
the height of the camera above the ground in meters,
the vertical resolution of the camera, the file extension of the images,
the vertical field of view of the camera and finally,
you must indicate if your dataset includes static objects that need to be removed in preprocessing.

A sample input to follow can be seen in _example.json_. Once this file is created,
you can run the system with the following command.

<pre>
python cropy_rows.py PATH_TO_INPUT_JSON
</pre>
