
# import keras
import keras
import os

# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.bin.train import create_models
import argparse
import sys
import keras_retinanet

from keras_maskrcnn.preprocessing import csv_generator
import pandas as pd

from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.callbacks import RedirectModel
from keras_maskrcnn.callbacks.eval import Evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
keras.backend.tensorflow_backend._get_available_gpus()

def create_callbacks(model, training_model, prediction_model, validation_generator,path_dir_weights, create_evaluation=Evaluate):
    callbacks = []

    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            path_dir_weights,
            '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone='resnet101',
                                                                dataset_type='pre-processed_CEM')
        ),
        verbose=1,
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        batch_size=1,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )
    callbacks.append(tensorboard_callback)

    # use prediction model for evaluation
    evaluation = create_evaluation(validation_generator, tensorboard=tensorboard_callback, weighted_average=True)

    evaluation = RedirectModel(evaluation, prediction_model)
    callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=2,
        min_lr=1e-7
    ))

    return callbacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code for training model to predict mask and class label')
    parser.add_argument('--path_annotations_train_csv', help='Path to the annotations for training (data needs to be pre-processed with preprocessing.utils.preprocessing)')
    parser.add_argument('--path_annotations_test_csv', help='Path to the annotations for testing (data needs to be pre-processed with preprocessing.utils.preprocessing)')
    parser.add_argument('--path_dir_weights', help='Path to save the weights')
    args = parser.parse_args()

    train_csv = args.path_annotations_train_csv
    test_csv = args.path_annotations_test_csv

    batch_size = 1

    transform_generator = random_transform_generator(
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0,
        flip_y_chance=0.5
    )

    train_generator = csv_generator.CSVGenerator(
        train_csv,
        transform_generator=transform_generator,
        csv_class_file='classes.csv',
        base_dir='.',
        batch_size=batch_size)
    validation_generator = csv_generator.CSVGenerator(
        test_csv,
        csv_class_file='classes.csv',
        base_dir='.',
        batch_size=batch_size)

    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    backbone = models.backbone("resnet101")
    weights_for_resnet101 = "ResNet-101-model.keras.h5"
    model, training_model, prediction_model = create_models(backbone_retinanet=backbone.maskrcnn, num_classes=2,
                                                            weights=weights_for_resnet101, freeze_backbone=False)

    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        args.path_dir_weights,
        validation_generator
    )

    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal(),
            'masks': keras_maskrcnn.losses.mask()
        },
        optimizer=keras.optimizers.Adam(lr=1e-5)
    )

    training_model.fit_generator(train_generator,
                        epochs=30,
                        verbose=1,
                        callbacks = callbacks,
                        steps_per_epoch=len(train_generator)//batch_size,
                        validation_data=validation_generator,
                        validation_steps = len(validation_generator)//batch_size)