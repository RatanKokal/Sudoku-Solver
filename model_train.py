import os, logging, cv2, argparse, matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from load_data import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Input

def train_model(saved_model, filename, epochs, batch_size, plot_path):

    # Load data
    training_data, validation_data = load_data(batch_size = batch_size, validation_split = 0.1)

    if saved_model:
        model = tf.keras.models.load_model(saved_model)
    else:
        # Create model
        model = Sequential()

        model.add(Input(shape = (28, 28, 1)))

        model.add(Conv2D(32, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(64, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(128, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation("relu"))

        model.add(Dense(128))
        model.add(Activation("relu"))

        model.add(Dense(32))
        model.add(Activation("relu"))

        model.add(Dense(10))
        model.add(Activation("softmax"))

        model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

    data = model.fit(training_data, epochs = epochs, batch_size = batch_size, validation_data = validation_data) 

    if plot_path:
        plt.plot(data.history['accuracy'], label='Training Accuracy')
        plt.plot(data.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(plot_path)

    # Save model
    model.save(filename)

def evaluate_model(model, batch_size, validation_split):

    # Load data
    training_data, validation_data = load_data(validation_split = validation_split, batch_size = batch_size)

    model = tf.keras.models.load_model(model)

    # Evaluate model
    loss, accuracy = model.evaluate(validation_data)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='command',
                                   title='subcommands',
                                   description='valid subcommands',
                                   help='additional help')
train = subparsers.add_parser('train')
evaluate = subparsers.add_parser('evaluate')

train.add_argument('--saved_model', 
                    help = 'Saved model to load and train further(default = None)',
                    type = str,
                    default = None)

train.add_argument('--save_as',
                    help = 'Save model as(default = \'./assets/models/model.keras\')',
                    type = str,
                    default = './assets/models/model.keras')

train.add_argument('--epochs',
                    help = 'Number of epochs(default = 15)',
                    type = int,
                    default = 15)

train.add_argument('--batch_size',
                    help = 'Batch size(default = 8)',
                    type = int,
                    default = 8)

train.add_argument('--plot',
                    help = 'Save the plot training and validation accuracy(default = None)',
                    type = str,
                    default = None)

evaluate.add_argument('--model',
                    help = 'Path of model to evaluate(default = \'./assets/models/model.keras\')',
                    type = str,
                    default = './assets/models/model.keras')

evaluate.add_argument('--batch_size',
                    help = 'Batch size(default = 8)',
                    type = int,
                    default = 8)

evaluate.add_argument('--fraction',
                    help = 'Fraction of data to evaluate (0, 1) (default = 0.1)',
                    type = float,
                    default = 0.1)

args = parser.parse_args()

if args.command == 'train':
    train_model(args.saved_model, args.save_as, args.epochs, args.batch_size, args.plot)
elif args.command == 'evaluate':
    evaluate_model(args.model, args.batch_size, args.fraction)
    



