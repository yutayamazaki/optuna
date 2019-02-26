"""
$ STORAGE_URL=sqlite:///example.db
$ STUDY_NAME=`optuna create-study --storage $STORAGE_URL`
$ mpirun -n 2 -- python horovod_integration.py $STUDY_NAME $STORAGE_URL
"""

from __future__ import division
from __future__ import print_function

import horovod.tensorflow as hvd
from mpi4py import MPI
import numpy as np
import shutil
import sys
import tensorflow as tf

import optuna

MODEL_DIR = "/tmp/mnist_convnet_model"
BATCH_SIZE = 100
TRAIN_STEPS = 1000
EVAL_STEPS = 100
PRUNING_INTERVAL_STEPS = 50


def create_network(trial, features):
    # We optimize the numbers of layers and their units.
    input_layer = tf.reshape(features['x'], [-1, 784])
    prev_layer = input_layer

    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        n_units = int(trial.suggest_int('n_units_l{}'.format(i), 1, 128))
        prev_layer = tf.layers.dense(inputs=prev_layer, units=n_units, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=prev_layer, units=10, activation=tf.nn.relu)
    return logits


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=adam_lr, weight_decay=weight_decay)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        momentum = trial.suggest_loguniform('momentum', 1e-5, 1e-1)
        optimizer = tf.contrib.opt.MomentumWOptimizer(
            learning_rate=momentum_sgd_lr, momentum=momentum, weight_decay=weight_decay)

    return hvd.DistributedOptimizer(optimizer)


def model_fn(trial, features, labels, mode):
    # Create network.
    logits = create_network(trial, features)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode).
        "classes": tf.argmax(input=logits, axis=1),

        # Add `softmax_tensor` to the graph. It is used for PREDICT.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes).
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = create_optimizer(trial)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode).
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def objective(trial, comm):
    # Load dataset.
    (train_data, train_labels), (eval_data, eval_labels) = \
        tf.keras.datasets.mnist.load_data('MNIST-data-{}'.format(hvd.rank()))

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = "{}/{}".format(MODEL_DIR, trial.trial_id) if hvd.rank() == 0 else None
    if model_dir is not None:
        # Delete old model.
        shutil.rmtree(model_dir, ignore_errors=True)

    # Create estimator.

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    config = tf.estimator.RunConfig(
        session_config=session_config,
        save_summary_steps=PRUNING_INTERVAL_STEPS,
        save_checkpoints_steps=PRUNING_INTERVAL_STEPS)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: model_fn(trial, features, labels, mode),
        model_dir=model_dir,
        config=config)

    # Setup pruning hook.
    optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
        trial=trial,
        estimator=mnist_classifier,
        metric="accuracy",
        is_higher_better=True,
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )

    # Train and evaluate the model.

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)

    # Horovod: adjust number of steps based on number of workers.
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=TRAIN_STEPS // hvd.size(),
        hooks=[bcast_hook, optuna_pruning_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=EVAL_STEPS, start_delay_secs=0, throttle_secs=0)

    eval_results, _ = tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
    return 1 - float(eval_results['accuracy'])


def main(unused_argv):
    # Horovod: initialize Horovod.
    hvd.init()

    study_name = sys.argv[1]
    storage_url = sys.argv[2]
    study = optuna.Study(study_name, storage_url)

    comm = MPI.COMM_WORLD
    mpi_study = optuna.integration.MPIStudy(study, comm)
    mpi_study.optimize(objective, n_trials=20)

    if comm.rank == 0:
        pruned_trials = [
            t for t in mpi_study.trials if t.state == optuna.structs.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in mpi_study.trials if t.state == optuna.structs.TrialState.COMPLETE
        ]
        print('Study statistics: ')
        print('  Number of finished trials: ', len(study.trials))
        print('  Number of pruned trials: ', len(pruned_trials))
        print('  Number of complete trials: ', len(complete_trials))

        print('Best trial:')
        trial = mpi_study.best_trial

        print('  Value: ', trial.value)

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))


if __name__ == "__main__":
    tf.app.run()
