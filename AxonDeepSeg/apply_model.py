import tensorflow as tf
import math
import os
import pickle
import numpy as np
import time
from mrf import run_mrf
from scipy import io
from scipy.misc import imread, imsave
from skimage.transform import rescale
from skimage import exposure
from sklearn import preprocessing
from config import*


def im2patches(img, size=256):
    """
    :param img: image to segment.
    :param size: size of the patches to extract (must be the same as used in the learning)
    :return: [img, patches, positions of the patches]
    """
    h, w = img.shape

    q_h, r_h = divmod(h, size)
    q_w, r_w = divmod(w, size)

    r2_h = size-r_h
    r2_w = size-r_w
    q2_h = q_h + 1
    q2_w = q_w + 1

    q3_h, r3_h = divmod(r2_h, q_h)
    q3_w, r3_w = divmod(r2_w, q_w)

    dataset = []
    positions=[]
    pos = 0
    while pos+size<=h:
        pos2 = 0
        while pos2+size<=w:
            patch = img[pos:pos+size, pos2:pos2+size]
            patch = exposure.equalize_hist(patch)
            patch = (patch - np.mean(patch))/np.std(patch)

            dataset.append(patch)
            positions.append([pos,pos2])
            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h

    return [img, np.asarray(dataset), positions]


def patches2im(predictions, positions, image_height, image_width):
    """
    :param predictions: list of the segmentation masks on the patches
    :param positions: positions of the segmentations masks
    :param h_size: height of the image to reconstruct
    :param w_size: width of the image to reconstruct
    :return: reconstructed segmentation on the full image from the masks and their positions
    """
    image = np.zeros((image_height, image_width))
    for pred, pos in zip(predictions, positions):
        image[pos[0]:pos[0]+256,pos[1]:pos[1]+256] = pred
    return image


def apply_convnet(path, model_path):
    """
    :param path: folder of the image to segment. Must contain image.jpg
    :param model_path: folder of the model of segmentation. Must contain model.ckpt
    :return: prediction, the mask of the segmentation
    """

    print '\n\n ---Start axon segmentation on %s---'%path

    path_img = path+'/image.jpg'
    img = imread(path_img, flatten=False, mode='L')

    file = open(path+'/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())

    # set the resolution to the general_pixel_size
    rescale_coeff = pixel_size/general_pixel_size
    img = (rescale(img,rescale_coeff)*256).astype(int)

    batch_size = 1
    depth = 6
    image_size = 256
    n_input = image_size * image_size
    n_classes = 2

    folder_model = model_path
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    if os.path.exists(folder_model+'/hyperparameters.pkl'):
        print 'hyperparameters detected in the model'
        hyperparameters = pickle.load(open(folder_model +'/hyperparameters.pkl', "rb"))
        depth = hyperparameters['depth']
        image_size = hyperparameters['image_size']

    x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
    y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
    keep_prob = tf.placeholder(tf.float32)


    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Create model
    def conv_net(x, weights, biases, dropout, image_size = image_size):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
        data_temp = x
        data_temp_size = [image_size]
        relu_results = []

    # contraction
        for i in range(depth):
          conv1 = conv2d(data_temp, weights['wc1'][i], biases['bc1'][i])
          conv2 = conv2d(conv1, weights['wc2'][i], biases['bc2'][i])
          relu_results.append(conv2)

          conv2 = maxpool2d(conv2, k=2)
          data_temp_size.append(data_temp_size[-1]/2)
          data_temp = conv2

        conv1 = conv2d(data_temp, weights['wb1'], biases['bb1'])
        conv2 = conv2d(conv1, weights['wb2'], biases['bb2'])
        data_temp_size.append(data_temp_size[-1])
        data_temp = conv2


    # expansion
        for i in range(depth):
            data_temp = tf.image.resize_images(data_temp, (data_temp_size[-1] * 2, data_temp_size[-1] * 2))
            upconv = conv2d(data_temp, weights['upconv'][i], biases['upconv'][i])
            data_temp_size.append(data_temp_size[-1]*2)

            upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[depth-i-1], [0, 0, 0, 0],
                                                                     [-1, data_temp_size[depth-i-1], data_temp_size[depth-i-1], -1]), upconv])
            conv1 = conv2d(upconv_concat, weights['we1'][i], biases['be1'][i])
            conv2 = conv2d(conv1, weights['we2'][i], biases['be2'][i])
            data_temp = conv2

        finalconv = tf.nn.conv2d(conv2, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
        final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], 2]))

        return final_result

    weights = {'wc1':[],'wc2':[],'we1':[],'we2':[],'upconv':[],'finalconv':[],'wb1':[], 'wb2':[]}
    biases = {'bc1':[],'bc2':[],'be1':[],'be2':[],'finalconv_b':[],'bb1':[], 'bb2':[],'upconv':[]}

    # Contraction
    for i in range(depth):
      if i == 0:
        num_features_init = 1
        num_features = 64
      else:
        num_features = num_features_init * 2


    # Store layers weight & bias

      weights['wc1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))), name = 'wc1-%s'%i))
      weights['wc2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name = 'wc2-%s'%i))
      biases['bc1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='bc1-%s'%i))
      biases['bc2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='bc2-%s'%i))

      image_size = image_size/2
      num_features_init = num_features
      num_features = num_features_init*2

    weights['wb1']= tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))),name='wb1-%s'%i)
    weights['wb2']= tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='wb2-%s'%i)
    biases['bb1']= tf.Variable(tf.random_normal([num_features]), name='bb2-%s'%i)
    biases['bb2']= tf.Variable(tf.random_normal([num_features]), name='bb2-%s'%i)

    num_features_init = num_features

    for i in range(depth):

        num_features = num_features_init/2
        weights['upconv'].append(tf.Variable(tf.random_normal([2, 2, num_features_init, num_features]), name='upconv-%s'%i))
        biases['upconv'].append(tf.Variable(tf.random_normal([num_features]), name='bupconv-%s'%i))
        weights['we1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))), name='we1-%s'%i))
        weights['we2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='we2-%s'%i))
        biases['be1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='be1-%s'%i))
        biases['be2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='be2-%s'%i))

        num_features_init = num_features

    weights['finalconv']= tf.Variable(tf.random_normal([1, 1, num_features, n_classes]), name='finalconv-%s'%i)
    biases['finalconv_b']= tf.Variable(tf.random_normal([n_classes]), name='bfinalconv-%s'%i)

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)


    saver = tf.train.Saver(tf.all_variables())

    # Image to batch
    image_init, data, positions = im2patches(img, 256)
    predictions = []

    # Launch the graph
    sess = tf.Session()
    saver.restore(sess, folder_model+"/model.ckpt")

    for i in range(len(data)):
        print 'processing patch %s on %s'%(i, len(data))
        batch_x = np.asarray([data[i]])
        start = time.time()
        p = sess.run(pred, feed_dict={x: batch_x})
        #print time.time() - start,'s - test time'
        prediction_m = p[:, 0].reshape(256,256)
        prediction = p[:, 1].reshape(256,256)

        Mask = prediction - prediction_m > 0
        predictions.append(Mask)

    sess.close()
    tf.reset_default_graph()

    h_size, w_size = image_init.shape
    prediction_rescaled = patches2im(predictions, positions, h_size, w_size)
    prediction = (preprocessing.binarize(rescale(prediction_rescaled, 1/rescale_coeff),threshold=0.5)).astype(int)

    return prediction

    #######################################################################################################################
    #                                            Mrf                                                                      #
    #######################################################################################################################

def axon_segmentation(image_path, model_path, mrf_path):
    """
    :param image_path: folder of the image to segment. Must contain image.jpg
    :param model_path: folder of the model of segmentation. Must contain model.ckpt
    :param mrf_path: folder of the mrf parameters.  Must contain mrf_parameter.pkl
    :return: no return

    Results including the prediction and the prediction associated with the mrf are saved in the image_path
    AxonMask.mat is saved in the image_path to feed the detection of Myelin
    /AxonSeg.jpeg is saved in the image_path

    """

    file = open(image_path+'/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())
    rescale_coeff = pixel_size/general_pixel_size


    # ------ Apply ConvNets ------- #
    prediction = apply_convnet(image_path, model_path)

    # ------ Apply mrf ------- #
    nb_class = 2
    max_map_iter = 10
    image_init = (rescale(imread(image_path+'/image.jpg', flatten=False, mode='L'),rescale_coeff)*256).astype(int)
    y_pred = (preprocessing.binarize((rescale(prediction.astype(float), rescale_coeff) * 256).astype(int), threshold=125)).reshape(-1, 1)

    folder_mrf = mrf_path
    mrf_parameters = pickle.load(open(folder_mrf +'/mrf_parameter.pkl', "rb"))
    weight = mrf_parameters['weight']

    img_mrf = run_mrf(y_pred, image_init, nb_class, max_map_iter, weight)
    img_mrf = img_mrf == 1
    img_mrf = preprocessing.binarize(rescale(img_mrf, 1/rescale_coeff),threshold=0.5)

    # ------ Saving results ------- #
    results={}
    results['img_mrf'] = img_mrf
    results['prediction'] = prediction

    with open(image_path+'/results.pkl', 'wb') as handle :
            pickle.dump(results, handle)

    imsave(image_path+'/AxonDeepSeg.jpeg', img_mrf, 'jpeg')

#---------------------------------------------------------------------------------------------------------

def myelin(path):
    """
    :param path: folder of the data, must include the segmentation results results.pkl
    :return: no return

    Myelin is Segmented by the AxonSegmentation Toolbox (NeuroPoly)
    The segmentation mask of the myelin is saved in the folder of the data
    """

    file = open(path+'/pixel_size_in_micrometer.txt', 'r')
    pixel_size = float(file.read())

    file = open(path+"/results.pkl",'r')
    results = pickle.load(file)


    print '\n\n ---START MYELIN DETECTION---'

    io.savemat(path+'/AxonMask.mat', mdict={'prediction': results["img_mrf"]})
    current_path = os.path.dirname(os.path.abspath(__file__))
    print current_path

    command = path_matlab+"/bin/matlab -nodisplay -nosplash -r \"clear all;addpath(\'"+current_path+"\');" \
            "addpath(genpath(\'"+path_axonseg+"/code\')); myelin(\'%s\',%s);exit()\""%(path,pixel_size)
    os.system(command)

    os.remove(path+'/AxonMask.mat')


def pipeline(image_path, model_path, mrf_path, visualize=False):
    """
    :param image_path: : folder of the data, must include image.jpg
    :param model_path :  folder of the model of segmentation. Must contain model.ckpt
    :param mrf_path: folder of the mrf parameters.  Must contain mrf_parameter.pkl
    :param visualize: if True, visualization of the results is runned. (and If a groundtruth is in image_path, scores are calculated)
    :return:
    """

    print '\n\n ---START AXON-MYELIN SEGMENTATION---'
    axon_segmentation(image_path, model_path, mrf_path)
    myelin(image_path)
    print '\n End of the process : see results in : ', image_path

    if visualize:
        from evaluation.visualization import visualize_results
        visualize_results(image_path)












