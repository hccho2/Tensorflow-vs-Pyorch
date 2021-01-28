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


## 3.3 Tensorflow `tf.keras.preprocessing.image_dataset_from_directory`


