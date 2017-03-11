"""Process an image dataset for faces
=====================================

Goes through a directory of images and copies over the ones with faces in them,
and record the detected bounding boxes and 5-point landmarks.

"""
from __future__ import print_function

import _init_paths

import argparse
import csv
import os
import shutil
import sys

import caffe
import cv2
import dlib
from scipy.misc import imread

from demo import detect_face


image_extensions = ['.png', '.jpg', '.jpeg']

csv_header = [
    'filename',
    'face_idx',
    'rect_x1',
    'rect_y1',
    'rect_x2',
    'rect_y2',
    'l_eye_x',
    'r_eye_x',
    'nose_x',
    'l_mouth_x',
    'r_mouth_x',
    'l_eye_y',
    'r_eye_y',
    'nose_y',
    'l_mouth_y',
    'r_mouth_y',
]



def get_infos(bbs, pts):
    assert bbs.shape[0] == pts.shape[0]
    num_ppl = bbs.shape[0]
    infos = list()
    for n in range(num_ppl):
        info = {
            'rect_x1':   bbs[n,0],
            'rect_y1':   bbs[n,1],
            'rect_x2':   bbs[n,2],
            'rect_y2':   bbs[n,3],
            'l_eye_x':   pts[n,0],
            'r_eye_x':   pts[n,1],
            'nose_x':    pts[n,2],
            'l_mouth_x': pts[n,3],
            'r_mouth_x': pts[n,4],
            'l_eye_y':   pts[n,5],
            'r_eye_y':   pts[n,6],
            'nose_y':    pts[n,7],
            'l_mouth_y': pts[n,8],
            'r_mouth_y': pts[n,9],
        }
        infos.append(info)
    return infos

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-m', '--metadata', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-cpu', action='store_true')
    args = parser.parse_args(argv)

    assert os.path.exists(args.input_dir)

    detector = dlib.get_frontal_face_detector()

    minsize = 60
    #threshold = [0.6, 0.7, 0.7]
    threshold = [0.6, 0.75, 0.9]
    factor = 0.709

    if args.use_cpu:
        caffe.set_mode_cpu()
    PNet = caffe.Net("model/det1.prototxt", "model/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net("model/det2.prototxt", "model/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net("model/det3.prototxt", "model/det3.caffemodel", caffe.TEST)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_file = open(args.metadata, 'w')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    csv_writer.writeheader()

    for root, _, filenames in os.walk(args.input_dir):
        imagenames = [x for x in filenames
                      if os.path.splitext(x)[-1] in image_extensions]
        for imagename in imagenames:
            directory = root.replace(args.input_dir, '')
            if directory:
                directory = directory[1:] # Get rid of the leading '/'
                dest_dir = os.path.join(args.output_dir, directory)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
            else:
                dest_dir = args.output_dir

            imagepath = os.path.join(root, imagename)


            img = imread(imagepath, mode='RGB')[:,:,::-1]


            dlib_faces = detector(img)
            if len(dlib_faces) == 0:
                print("No dlib face detected in", directory+imagename)
                continue

            bbs, pts = detect_face(img, minsize, PNet, RNet, ONet,
                                  threshold, False, factor)
            if len(bbs) > 0:
                infos = get_infos(bbs, pts)

                for idx, info in enumerate(infos):
                    info['filename'] = os.path.join(directory, imagename)
                    info['face_idx'] = str(idx)
                    csv_writer.writerow(info)

                shutil.copyfile(imagepath, os.path.join(dest_dir, imagename))

                if args.debug:
                    img = cv2.imread(imagepath)
                    for x in infos:
                        rect1 = (int(x['rect_x1']), int(x['rect_y1']))
                        rect2 = (int(x['rect_x2']), int(x['rect_y2']))
                        cv2.rectangle(img, rect1, rect2, (0,255,0), 1)
                        cv2.circle(img, (x['l_eye_x'],   x['l_eye_y']),   2, (255,  0,  0), 1, 8, 0)
                        cv2.circle(img, (x['r_eye_x'],   x['r_eye_y']),   2, (255,255,  0), 1, 8, 0)
                        cv2.circle(img, (x['nose_x'],    x['nose_y']),    2, (  0,255,  0), 1, 8, 0)
                        cv2.circle(img, (x['l_mouth_x'], x['l_mouth_y']), 2, (  0,255,255), 1, 8, 0)
                        cv2.circle(img, (x['r_mouth_x'], x['r_mouth_y']), 2, (  0,  0,255), 1, 8, 0)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
            else:
                print("No MTCNN face detected in", directory+imagename)

    if args.debug:
        cv2.destroyAllWindows()

    csv_file.close()


if __name__ == '__main__':
    main(sys.argv[1:])
