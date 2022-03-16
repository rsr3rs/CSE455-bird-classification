import matplotlib.pyplot as plt

f = open("acc.txt", "r")
train_epochs = []
test_epochs = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for l in f:
    component = l.strip().split(", ")
    if len(component) < 4:
        continue
    epoch = int(component[0])
    is_train = component[1] == 'train'
    loss = float(component[-2])
    acc = float(component[-1])
    if is_train:
        train_epochs.append(epoch)
        train_loss.append(loss)
        train_acc.append(acc)
    else:
        test_epochs.append(epoch)
        test_loss.append(loss)
        test_acc.append(acc)

plt.plot(train_epochs, train_loss)
plt.plot(test_epochs, test_loss)
plt.legend(["Train loss", "Test loss"])
plt.show()
plt.plot(train_epochs, train_acc)
plt.plot(test_epochs, test_acc)
plt.legend(["Train acc", "Test acc"])
plt.show()

