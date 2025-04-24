from zipfile import ZipFile
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
    
    
def unzip(folder_name):
    with ZipFile(folder_name, 'r') as myzip:
        myzip.extractall()


  
def walk_on(data_dir):
    for dirpath,dirname,filenames in  os.walk(data_dir):
        print(f"There are {dirname} directories and {len(filenames)} images in '{dirpath}'.")
        



def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = f"{dir_name}/{experiment_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print("Saved tensorboard log files to",log_dir)
    return tensorboard_callback



def loss_and_accuracy_plot(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  epochs = range(len(history.history['loss']))
  plt.figure(figsize=(15,7))
  plt.subplot(1,2,1)
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('accuracy')
  plt.xlabel('epochs')
  plt.legend()
  plt.show()
  
  
  
def compare_history(feature_extractors_history, fine_tuning_history, initial_epoch = 5):

    acc = feature_extractors_history.history['accuracy']
    loss = feature_extractors_history.history['loss']


    val_acc = feature_extractors_history.history['val_accuracy']
    val_loss = feature_extractors_history.history['val_loss']

    total_acc = acc +  fine_tuning_history.history['accuracy']
    total_loss = loss + fine_tuning_history.history['loss']


    total_val_acc =val_acc+ fine_tuning_history.history['val_accuracy']
    total_val_loss =val_loss + fine_tuning_history.history['val_loss']



    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc,label="Training Accuracy")
    plt.plot(total_val_acc, label ="Validatain Accuracy")

    plt.plot([initial_epoch-1,initial_epoch-1],
          plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')


    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epoch-1,initial_epoch-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()
