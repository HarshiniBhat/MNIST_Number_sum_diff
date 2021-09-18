# MNIST_Number_sum_Difference

To Write a neural network that can:
take 2 inputs:
- An image from MNIST dataset
- A random number between 0 and 9
 gives three outputs:
- the number that was represented by the MNIST image
- the sum 
- the difference of the number (predicted) from the MNIST dataset and the random number that was generated and sent as the input to the network.
## Approach to the problem

- Importing the required libraries

- Downloading the data
- Defining a dataset class
- Using the defined class and getting the data into the train_loader, test_loader
- Visualising the data.
- Defining the network
- Training the model on the defined network

## Downloading the data 

```python
# For training set 
train_set = torchvision.datasets.MNIST(
    root = './data', #creating directory and giving the path
    train = True, # as we are downloading the training set
    download = True,# True id the data is not present/available in local storage
    transform = transforms.Compose([
          transforms.ToTensor(), # converting image to tensor
          transforms.Normalize((0.1307,),(0.3081,)) #Normalizing the image with mean and std deviation
# For test set 
test_set = torchvision.datasets.MNIST(
    root = './data', #creating directory and giving the path
    train = False, # as we are downloading the training set
    download = True,# True id the data is not present/available in local storage
    transform = transforms.Compose([
          transforms.ToTensor(), # converting image to tensor
          transforms.Normalize((0.1307,),(0.3081,)) #Normalizing the image with mean and std deviation
```
Downloading MNIST dataset from torchvision directory and storing it in train_set and test_set

  ## Defining a project class

  ```python
  class Project(Dataset):
  def __init__(self, data):
    self.data = data
    self.rand_num = torch.randint(0,10,(len(data),1), dtype =torch.float)
  
  def __getitem__(self, index):
    image, label = self.data[index]
    num = self.rand_num[index]

    sum = num + label
    diff = abs(num - label)
    return image, num, label, sum, diff
  
  def __len__(self):
    return len(self.data)
```
The class is used to generate a random number and to
give the input image from the dataset,it returns
the image,label, number, sum, difference.
```python
 ## Using the defined class and creating an iterable object train_loader and test_loader
 rain_loader = torch.utils.data.DataLoader(
    Project(train_set), batch_size=batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    Project(test_set), batch_size=batch_size, shuffle=True, **kwargs
)
```
 ## Visualising the data
```python
cols, rows = 4,2
figure = plt.figure(figsize = (10,8))
for i in range(1,cols*rows +1): # for loop for number of elements in the final output
  k = np.random.randint(0,batch_size) # to generate a random number within the batch_size

  figure.add_subplot(rows, cols, i) # adding subplot for each image
  plt.title(f" Label: {label[k].item()}, \nRandom number: {num[k].item()}, \nsum: {sum[k].item()}, diff: {diff[k].item()}")
  plt.imshow(image[k].squeeze(), cmap = 'gray')
  plt.axis('off')
```
Visualising the image with its label and sum, difference
## Defining the network

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input 28*28*1, kernel size -3*3*1, no of kernels 32,  OUtput 28*28*32 LRF =3 GRF =5
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)#input 28*28*32, kernel size -3*3*32, no of kernels 64,  OUtput 28*28*64 LRF =3 GRF =7
        self.pool1 = nn.MaxPool2d(2, 2)#input 28*28*64, kernel size -2*2, no of kernels 64, OUtput 14*14*64 LRF =2 GRF =14
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)#input 14*14*64, kernel size -3*3*64, no of kernels 128,  OUtput 14*14*128,lRF =3 GRF = 16
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)#input 14*14*128, kernel size -3*3*128, no of kernels 256,  OUtput 14*14*256, LRF =3 GRF =18
        self.pool2 = nn.MaxPool2d(2, 2)#input 14*14*256, kernel size -2*2, no of kernels 256, OUtput 7*7*256 LRF =3 GRF =36
        self.conv5 = nn.Conv2d(256, 512, 3)#input 7*7*256, kernel size -3*3*256, no of kernels 512 ,OUtput 5*5*512 LRF =3 GRF =38
        self.conv6 = nn.Conv2d(512, 1024, 3)#input 5*5*512, kernel size -3*3*512, no of kernels 1024 ,OUtput3*3*1024 LRF =3 GRF =40
        self.conv7 = nn.Conv2d(1024, 10, 3)#input 3*3*1024, kernel size -3*3*1024, no of kernels 10 ,OUtput1*1*10 LRF =3 GRF =42

        self.dense1 = nn.Linear(1,16)
        self.dense2 = nn.Linear(16,32)
        
        #sum
        self.sum1 = nn.Linear(10+32,64)
        self.sum2 = nn.Linear(64,128)
        self.sum3 = nn.Linear(128,1)

        #diff
        self.diff1 = nn.Linear(10+32,64)
        self.diff2 = nn.Linear(64,128)
        self.diff3 = nn.Linear(128,1)

    def forward(self, x, num):# calling defined objects through forward function.
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))) # 1 convolutional block
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)# Flattening of the 1*1*10 matrix 

        num = F.relu(self.dense1(num))
        num = F.relu(self.dense2(num))
        num = torch.cat((x,num), dim =1)
        #sum processor
        out_sum = F.relu(self.sum1(num))
        out_sum = F.relu(self.sum2(out_sum))
        out_sum = self.sum3(out_sum)

        #diff processor
        out_diff = F.relu(self.diff1(num))
        out_diff = F.relu(self.diff2(out_diff))
        out_diff = self.diff3(out_diff)

        return x, out_sum, out_diff 
```
we have a network with 7 convolution  layers and two dense 
layers and 3 linear layers for sum and difference .
 we also used max pooling, activation function ReLU to reduce the effect 
 of vanishing gradient 
## Training the Module
```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (images, nums, labels, sums, diffs) in enumerate(pbar):
        images, nums, labels, sums, diffs = images.to(device), nums.to(device), labels.to(device), sums.to(device), diffs.to(device)
        optimizer.zero_grad()
        label, sum, diff = model(images, nums)

        loss1 = F.nll_loss(F.softmax(label,dim =1), labels)
        loss2 = F.mse_loss(sum,sums)
        loss3 = F.mse_loss(diff,diffs)

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, nums, labels, sums, diffs in test_loader:
            images, nums, labels, sums, diffs = images.to(device), nums.to(device), labels.to(device), sums.to(device), diffs.to(device)
            label, sum, diff = model(images, nums)

            loss1 = F.nll_loss(F.softmax(label), labels, reduction='sum')
            loss2 = F.mse_loss(sum,sums, reduction='sum')
            loss3 = F.mse_loss(diff,diffs, reduction='sum')
            loss = loss1 + loss2 +loss3

            test_loss += loss.item()  # sum up batch loss
            pred = label.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)   
```
we calculate the loss and evaluate the model.




  