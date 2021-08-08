import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation="relu", padding="same", name="Input", input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation="relu", padding="same", name="PreHandle"))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation="relu", padding="same", name="Handle"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="MaxPooling"))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(5,5), activation="relu", padding="same", name="Final"))
model.add(tf.keras.layers.Flatten(name="Flatten"))
model.add(tf.keras.layers.Dense(100, activation="relu", name="FullChain"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation="softmax", name="Result"))
model.summary()
model.load_weights("Model.h5")
print("model loaded!")
model.summary()

def load_data_npz(path="mnist.npz"):
    f = np.load(path)
    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    f.close()
    return x_train,y_train,x_test,y_test

X_train, Y_train, X_test, Y_test = load_data_npz()
print("x_train:{}".format(X_train.shape))
print("y_train:{}".format(Y_train.shape))
print("x_test:{}".format(X_test.shape))
print("y_test:{}".format(Y_test.shape))

print(X_train[0],Y_train[0])

train_x,train_y = [],[]
for i in range(len(X_train)):
    temp = X_train[i].astype(np.float32)/255
    temp = temp.reshape(28,28,1)
    train_x.append(temp),train_y.append(Y_train[i])
test_x,test_y = [],[]
for i in range(len(X_test)):
    temp = X_test[i].astype(np.float32)/255
    temp = temp.reshape(28,28,1)
    test_x.append(temp),test_y.append(Y_test[i])

train_x = (np.array(train_x))
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_x = (np.array(test_x))
test_y = tf.keras.utils.to_categorical(test_y, 10)
print("img handled!")
print(test_x.shape)

img = X_train[0]
#img = image.load_img(img_path,target_size=(224,224))
# #显示图片
# plt.imshow(img)
# plt.show()
x = image.img_to_array(img)
#维度扩展
x = np.expand_dims(x,axis=0)
#图片预处理
# preds = model.predict(x)
# print('predicted', decode_predictions(preds, top=3)[0])
layer_names = ['PreHandle','Handle','Final']
#layer_names = ['block1_conv1','block3_conv1','block5_conv1']
#获取指定层的输出
layer_outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
#获得模型指定层的输出
activation_model = keras.models.Model(inputs=model.input,outputs=layer_outputs)
#获得输出
activations = activation_model.predict(x)
first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0,:,:,1],cmap='viridis')
plt.show()

images_per_row = 8

for layer_name,layer_activation in zip(layer_names,activations):
    #获取卷积核的个数
    n_features = layer_activation.shape[-1]
    #特征图的形状(1,size,size,n_features)
    size = layer_activation.shape[1]

    n_cols = 1 # images_per_row
    #n_cols = 8
    print("this is the n_cols")
    print(n_cols)
    display_grid = np.zeros((size * n_cols,images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:,col * images_per_row + row]
            #归一化
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 128
            channel_image += 128
            channel_image = np.clip(channel_image,0,255).astype(np.uint8)
            display_grid[col * size : (col + 1) * size,row * size: (row + 1) * size] = channel_image
        
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
    plt.subplots_adjust(wspace=1,hspace=1)
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid,aspect='auto',cmap='viridis')
plt.show()