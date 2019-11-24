from fractal_analysis import FractalReport as FR
from flask import (Flask, render_template, request,
                    redirect, session, url_for)
from flask_dropzone import Dropzone
from flask_uploads import (UploadSet, configure_uploads, IMAGES,
                             patch_request_class)

import os

app = Flask(__name__)

dropzone = Dropzone(app)

# Dropzone settings

app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
app.config['SECRET_KEY'] = 'supersecretkey'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

# global report list
report_urls = []

@app.route('/', methods=['GET', 'POST'])
def index():

    # setting session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold uploaded image urls
    file_urls = session['file_urls']

    # list to hold uploaded image urls
    file_urls = []

    
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            image_file = request.files.get(f)
            print(image_file.filename)

            # save the file to the 'photos' folder
            filename = photos.save(image_file, name=image_file.filename)
            print("*** DEBUG:", filename,'***')

            print("*** DEBUG:", 'uploads'+'\\'+filename,'***')
            # generating FRACTAL ANALYSIS images
            fr = FR.FractalReport('uploads'+'\\'+filename, 'static')

            fr.hsv_graph_3d()
            fr.otsu_thresh_image()
            fr.fractal_transformation()

            # append image urls
            file_urls.append(photos.url(filename))

            report_urls.extend(fr.im_report_paths_list)

        session['file_urls'] = file_urls
        return "uploading..."
    
    # return dropzone template on GET request
    return render_template('index.html')

@app.route('/results')
def results():

    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))

    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)

    return render_template('results.html', file_urls=file_urls, report_urls=report_urls)