from __future__ import division
import random
import sys
import time
import numpy as np
import pickle

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators

import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn import losses as losses_f

sys.setrecursionlimit(40000)

def run(options, dataset):
	C = config.Config()

	C.model_path = options['output_weight_path']
	C.num_rois = int(options['num_rois'])

	if options['network'] == 'vgg':
		C.network = 'vgg'
		from keras_frcnn import vgg as nn
	elif options['network'] == 'resnet50':
		from keras_frcnn import resnet as nn
		C.network = 'resnet50'
	else:
		print('Not a valid model')
		raise ValueError

	# check if weight path was passed via command line
	if 'input_weight_path' in options:
		C.base_net_weights = options['input_weight_path']
	else:
		# set the path to weights based on backend and model
		C.base_net_weights = nn.get_weight_path()

	config_output_filename = options['config_filename']

	with open(config_output_filename, 'wb') as config_f:
		pickle.dump(C,config_f)
		print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

	data_gen_train = data_generators.get_anchor_gt(dataset, C, nn.get_img_output_length, K.image_dim_ordering())

	model_all, model_classifier, model_rpn = construct_models(C)

	epoch_length = 1000
	num_epochs = int(options['num_epochs'])
	iter_num = 0

	losses = np.zeros((epoch_length, 5))
	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []
	start_time = time.time()

	best_loss = np.Inf

	print('Starting training')

	for epoch_num in range(num_epochs):

		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

		while True:
			try:

				if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []
					print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
					if mean_overlapping_bboxes == 0:
						print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

				X, Y, bboxes = next(data_gen_train)

				loss_rpn = model_rpn.train_on_batch(X, Y)

				P_rpn = model_rpn.predict_on_batch(X)

				R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

				# note: Calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
				X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, dict(bboxes=bboxes, width=X.shape[1], height=X.shape[0]), C, {'car':0, 'bg':1})

				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue

				sel_samples = select_roi_samples(C, Y1, rpn_accuracy_for_epoch, rpn_accuracy_rpn_monitor)

				loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

				record_losses(iter_num, loss_class, loss_rpn, losses)

				iter_num += 1

				progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
										  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

				if iter_num == epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
					rpn_accuracy_for_epoch = []

					if C.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))
						print('Elapsed time: {}'.format(time.time() - start_time))

					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					iter_num = 0
					start_time = time.time()

					if curr_loss < best_loss:
						if C.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
						best_loss = curr_loss
						model_all.save_weights(C.model_path)

					break

			except Exception as e:
				continue

	print('Training complete, exiting.')


def record_losses(iter_num, loss_class, loss_rpn, losses):
	losses[iter_num, 0] = loss_rpn[1]
	losses[iter_num, 1] = loss_rpn[2]
	losses[iter_num, 2] = loss_class[1]
	losses[iter_num, 3] = loss_class[2]
	losses[iter_num, 4] = loss_class[3]


def select_roi_samples(C, Y1, rpn_accuracy_for_epoch, rpn_accuracy_rpn_monitor):
	neg_samples = np.where(Y1[0, :, -1] == 1)
	pos_samples = np.where(Y1[0, :, -1] == 0)
	if len(neg_samples) > 0:
		neg_samples = neg_samples[0]
	else:
		neg_samples = []
	if len(pos_samples) > 0:
		pos_samples = pos_samples[0]
	else:
		pos_samples = []
	rpn_accuracy_rpn_monitor.append(len(pos_samples))
	rpn_accuracy_for_epoch.append((len(pos_samples)))
	if C.num_rois > 1:
		if len(pos_samples) < C.num_rois // 2:
			selected_pos_samples = pos_samples.tolist()
		else:
			selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
		try:
			selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
													replace=False).tolist()
		except:
			selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
													replace=True).tolist()

		sel_samples = selected_pos_samples + selected_neg_samples
	else:
		# in the extreme case where num_rois = 1, we pick a random pos or neg sample
		selected_pos_samples = pos_samples.tolist()
		selected_neg_samples = neg_samples.tolist()
		if np.random.randint(0, 2):
			sel_samples = random.choice(neg_samples)
		else:
			sel_samples = random.choice(pos_samples)
	return sel_samples


def construct_models(C):
	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
	else:
		input_shape_img = (None, None, 3)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(None, 4))
	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)
	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn = nn.rpn(shared_layers, num_anchors)
	classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=2, trainable=True)
	model_rpn = Model(img_input, rpn[:2])
	model_classifier = Model([img_input, roi_input], classifier)
	# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
	model_all = Model([img_input, roi_input], rpn[:2] + classifier)

	try:
		print('loading weights from {}'.format(C.base_net_weights))
		model_rpn.load_weights(C.base_net_weights, by_name=True)
		model_classifier.load_weights(C.base_net_weights, by_name=True)
	except:
		print('Could not load pretrained model weights. Weights can be found in the keras application folder \
			https://github.com/fchollet/keras/tree/master/keras/applications')

	optimizer = Adam(lr=1e-5)
	optimizer_classifier = Adam(lr=1e-5)
	model_rpn.compile(optimizer=optimizer,
					  loss=[losses_f.rpn_loss_cls(num_anchors), losses_f.rpn_loss_regr(num_anchors)])
	model_classifier.compile(optimizer=optimizer_classifier,
							 loss=[losses_f.class_loss_cls, losses_f.class_loss_regr(1)],
							 metrics={'dense_class_{}'.format(2): 'accuracy'})
	model_all.compile(optimizer='sgd', loss='mae')

	return model_all, model_classifier, model_rpn