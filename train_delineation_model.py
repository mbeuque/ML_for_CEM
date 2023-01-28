
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

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = 'weights_model.h5'


# load label to names mapping for visualization purposes
labels_to_names = {0:'benign',1:'malignant'}

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

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
    transform_generator = transform_generator,
    csv_class_file=r'E:\ManonData\keras-retinanet-master\classes.csv',
    base_dir='.',
    batch_size=batch_size)
validation_generator = csv_generator.CSVGenerator(
    test_csv,
    csv_class_file=r'E:\ManonData\keras-retinanet-master\classes.csv',
    base_dir='.',
    batch_size=batch_size)

backbone = models.backbone("resnet101")
weights_for_resnet101 = "ResNet-101-model.keras.h5"
model, training_model, prediction_model = create_models(backbone_retinanet=backbone.maskrcnn, num_classes=2, weights=weights_for_resnet101, freeze_backbone=False)


def create_callbacks_v2(model, training_model, prediction_model, validation_generator, create_evaluation=Evaluate):
    callbacks = []

    # save the model
    snapshot_path = 'weights_mask_rcnn'
    # ensure directory created first; otherwise h5py will error after epoch.
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            snapshot_path,
            '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone='resnet101',
                                                                dataset_type='pre-processed_sergey')
        ),
        verbose=1,
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    tensorboard_callback = None

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
    train_csv = "annotations_train_dataset.csv"
    test_csv = "annotations_test_dataset.csv"
    external_df = pd.read_csv("external_validation.csv")

    callbacks = create_callbacks_v2(
        model,
        training_model,
        prediction_model,
        validation_generator,
        #setup args see setup gpus
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
                        steps_per_epoch=1507//batch_size,
                        validation_data=validation_generator,
                        #use_multiprocessing=True,
                        validation_steps = 391//batch_size)