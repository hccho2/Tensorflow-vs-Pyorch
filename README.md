# Tensorflow vs Pyorch

## 1. Simple RNN Model
### 1.1 Tensorflow
```
# sincurve fitting

hidden_dim = 10
model = tf.keras.models.Sequential([ 
tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(hidden_dim),return_sequences=True,return_state=False), tf.keras.layers.Dense(1)])

N = 1000

x = np.linspace(0, 2 * np.pi, N)
y = np.sin(x) + np.random.uniform(-0.05, 0.05, size=x.shape)

data_x = np.array(y[:-1])
data_y = np.array(y[1:])


num_iter = 500
T= 5
optimizer = tf.keras.optimizers.Adam(lr=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()
train_loss = []
for i in range(num_iter):
    random_pos = np.random.randint(0,len(data_x)-T)
    input_x = data_x[random_pos:random_pos+T].reshape(1,T,1)
    target = data_y[random_pos:random_pos+T].reshape(1,T,1)

    with tf.GradientTape() as tape:
        pred = model(input_x)
        loss = loss_fn(pred,target)

    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.append(loss.numpy())

    if i% 20 ==0:
        print(f'{i}: loss: {loss}')

plt.plot(train_loss)
plt.show()

pred = model(data_x.reshape(1,-1,1))  # 

plt.plot(data_y,label='target',linewidth=5)
plt.plot(pred.numpy().reshape(-1),label='pred')

plt.legend()
plt.show()
```
### 1.2 Pytorch
```
# sincurve fitting

class MyRNN(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1,batch_first=True )
        self.fc = nn.Linear(hidden_dim,1)
    def forward(self,x):
        x,h = self.rnn(x)
        out = self.fc(x)
        return out
hidden_dim = 10
model = MyRNN(hidden_dim)

N = 1000

x = np.linspace(0, 2 * np.pi, N)
y = np.sin(x) + np.random.uniform(-0.05, 0.05, size=x.shape)

data_x = torch.Tensor(np.array(y[:-1]))
data_y = torch.Tensor(np.array(y[1:]))


num_iter = 500
T= 5
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
train_loss = []
for i in range(num_iter):
    optimizer.zero_grad()

    random_pos = np.random.randint(0,len(data_x)-T)
    input_x = data_x[random_pos:random_pos+T].reshape(1,T,1)
    target = data_y[random_pos:random_pos+T].reshape(1,T,1)

    pred = model(input_x)
    loss = loss_fn(pred,target)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.detach().numpy())

    if i% 20 ==0:
        print(f'{i}: loss: {loss}')

plt.plot(train_loss)
plt.show()

pred = model(data_x.reshape(1,-1,1))  # 

plt.plot(data_y,label='target',linewidth=5)
plt.plot(pred.detach().numpy().reshape(-1),label='pred')

plt.legend()
plt.show()

```

## 2. Data Loading
### 2.1 Tensorflow: `tf.keras.utils.Sequence`
```
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([resize(imread(file_name), (200, 200)) for file_name in batch_x]), np.array(batch_y)
```

### 2.2 Pytorch: `torch.utils.data.Dataset, DataLoader`

## 3. Image Dataset Loading
## 3.1 Pytorch `torchvision.datasets.ImageFolder`
```
import torchvision
import torchvision.datasets as datasets
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15,15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

data_dir = r'D:\hccho\CommonDataset\hymenoptera_data\small'   # 테스트를 위해, data몇개만 모아, 작은 dataset을 만듬.

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset =  datasets.ImageFolder(data_dir, data_transforms)
class_names = train_dataset.classes
print(class_names)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5,shuffle=True)

for i in range(3):
    inputs, classes = next(iter(dataloader))

    out = torchvision.utils.make_grid(inputs)  # inputs: 5, 3, 224, 224  ---> out: 3, 228, 1132
    imshow(out, title=[class_names[x] for x in classes])
```
<p align="center"><img src="torchvision_result.png" />  </p>

## 3.2 Tensorflow `tf.keras.preprocessing.image import ImageDataGenerator`
```
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_dir = './small'
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

#class_mode='categorical'  ==> label을 onehot으로 만들어서 return한다.
train_generator = train_datagen.flow_from_directory(data_dir, target_size=(150, 150), batch_size=8,class_mode='categorical',shuffle=False)

print('class name: ', train_generator.class_indices)  # list(train_generator.class_indices) ==> ['ants', 'bees']
print('batch_size = ',  train_generator.batch_size, 'image shape: ', train_generator.image_shape)

class_names = list(train_generator.class_indices.keys())
for i in range(3):
    inputs, classes = next(iter(train_generator))
    inputs = np.concatenate(inputs, axis=1)  # (N,150,150,3)  ==> (150,750,3)
    classes = classes.argmax(axis=-1)
    plt.figure(figsize=(15,35))
    plt.imshow(inputs)
    plt.title([class_names[x] for x in classes])
    plt.show()

```
<p align="center"><img src="ImageDataGenerator_result.png" />  </p>

## 3.3 Tensorflow `tf.keras.preprocessing.image_dataset_from_directory`
```
def my_preprocessing(image,label):

    image = image/255.   # tf.image.convert_image_dtype는 정수가 들어 왔을 때, 0~1로 변환한다. 넘어온 image에는 resize되면서 0~255사이의 float 값이 들어 있다.

    #image = tf.image.random_flip_left_right(image)  # 확률 50%로 고정되어 있음.
    #image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 채도 조절
    # random crop
    shape = tf.shape(image)  # batch size 알아내기
    #image = tf.image.resize(image, (180,180))
    image = tf.image.random_crop(image,(shape[0],128,128,3))

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

ds = tf.keras.preprocessing.image_dataset_from_directory('./small',class_names=None, color_mode='rgb', batch_size=8, image_size=(150,150), shuffle=False)
class_names = ds.class_names
ds = ds.map(my_preprocessing)
ds = ds.repeat(5)

it = iter(ds)
for i in range(3):
    x, y = next(it)
    x = np.concatenate(x.numpy(), axis=1)  # (N,150,150,3)  ==> (150,750,3)
    plt.figure(figsize=(15,35))
    plt.imshow(x)
    plt.title([class_names[i] for i in y])
    plt.show()

```
<p align="center"><img src="image_dataset_from_directory_result.png" />  </p>
