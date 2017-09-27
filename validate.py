import eval_challenge
import test_frcnn
from datasets import Nexar2ValidationDataset


if __name__ == '__main__':

    dataset = Nexar2ValidationDataset('/Users/fs/Documents/Code/keras-frcnn/train_data/train_boxes.csv',
                                      '/Users/fs/Documents/Code/keras-frcnn/train_data/img',
                                      './val_filenames_test.txt')

    validation_dt_path = './validation_eval_input.csv'
    validation_gt_path = './validation_eval_gt.csv'

    options = {
        'num_rois': 32,
        'network': 'resnet50',
        'config_filename': './config.pickle',
        'bboxes_output': validation_dt_path
    }

    test_frcnn.run_test(options, dataset)

    dataset.generate_gt_file(validation_gt_path)

    iou_threshold = 0.75
    print ('{}'.format(eval_challenge.eval_detector_csv(validation_gt_path,
                                                        validation_dt_path,
                                                        iou_threshold)))