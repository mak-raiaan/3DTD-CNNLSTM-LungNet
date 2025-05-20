import os
import argparse
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_model(input_shape=(224, 224, 3, 1)):
    """Build the CNN-LSTM model architecture."""
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, 5, activation='relu', padding='same', dilation_rate=(1, 1), name='conv1'), 
                             input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')))
    model.add(TimeDistributed(Conv2D(32, 5, activation='relu', padding='same', dilation_rate=(2, 2), name='conv2')))
    model.add(TimeDistributed(Conv2D(64, 5, activation='relu', padding='same', dilation_rate=(3, 3), name='conv3')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')))
    model.add(TimeDistributed(Conv2D(32, 5, activation='relu', padding='same', dilation_rate=(4, 4), name='conv4')))
    model.add(TimeDistributed(Conv2D(64, 5, activation='relu', padding='same', dilation_rate=(5, 5), name='conv5')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(4, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_data_generators(train_dir, val_dir, batch_size=32, img_size=(224, 224)):
    """Create data generators for training and validation datasets."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return training_set, val_set

def train_model(model, training_set, val_set, epochs=100, weights_path='weights', model_path='model'):
    """Train the model and save weights and model."""
    # Create directories if they don't exist
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Setup callbacks
    weight_file = os.path.join(weights_path, 'best_weights.h5')
    log_file = os.path.join(weights_path, 'training_log.csv')
    
    checkpoint = ModelCheckpoint(
        weight_file, 
        monitor='val_accuracy', 
        verbose=1,
        save_weights_only=True,
        save_best_only=True, 
        mode='max'
    )
    
    log_csv = CSVLogger(log_file, separator=',', append=False)
    callbacks_list = [checkpoint, log_csv]
    
    # Train the model
    history = model.fit(
        training_set,
        epochs=epochs,
        validation_data=val_set,
        steps_per_epoch=len(training_set),
        validation_steps=len(val_set),
        callbacks=callbacks_list,
        shuffle=False
    )
    
    # Save the complete model
    model_file = os.path.join(model_path, 'video_frame_classifier.h5')
    model.save(model_file)
    print(f"Model saved to {model_file}")
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Train a CNN-LSTM model on video frames.")
    parser.add_argument('--train', required=True, help='Path to training data directory')
    parser.add_argument('--val', required=True, help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--weights_path', default='weights', help='Directory to save model weights')
    parser.add_argument('--model_path', default='model', help='Directory to save the complete model')
    
    args = parser.parse_args()
    
    # Create data generators
    print("Creating data generators...")
    training_set, val_set = create_data_generators(
        args.train, 
        args.val,
        batch_size=args.batch_size
    )
    
    # Build the model
    print("Building model...")
    model = build_model()
    model.summary()
    
    # Train the model
    print("Training model...")
    train_model(
        model, 
        training_set, 
        val_set, 
        epochs=args.epochs,
        weights_path=args.weights_path,
        model_path=args.model_path
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
