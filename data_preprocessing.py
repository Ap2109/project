from keras.src.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('C:\\Users\\manis\\PycharmProjects\\plant1\\Generated_dataset\\train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('C:\\Users\\manis\\PycharmProjects\\plant1\\Generated_dataset\\test', target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory('C:\\Users\\manis\\PycharmProjects\\plant1\\Generated_dataset\\val', target_size=(224, 224), batch_size=32, class_mode='categorical')