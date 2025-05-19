import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#########  load data ###########################
synthetic_lc = np.loadtxt('synthetic_lc.dat')
spots_par = np.loadtxt('spot_par.dat')
n_sample = spots_par.shape[0]
x_t = synthetic_lc[:,2]  ### x_t is a vector of magnitude values
print(x_t.shape)
x_t = x_t.reshape(n_sample,-1)  ### n_sample synthetic curves
print(x_t.shape)
y_t = spots_par[:,1:5]  #### 4 spots parameters as target values
print(y_t.shape)

for i in range(n_sample):
    m_x_t = np.mean(x_t[i,:])
    std_x_t = np.std(x_t[i,:])
    x_t[i,:] = (x_t[i,:] - m_x_t) / std_x_t

## for uniform distribution min-max scaling (min-max)
## for normal distribution Z scaling (mean-std)
scale_par_1 = [] # scale parameter 1 (mean or max_val)
scale_par_2 = [] # scale parameter 2 (std or min_val)
for j in range(4):
    if j <= 2:
        max_val = np.max(y_t[:,j])
        min_val = np.min(y_t[:,j])
        scale_par_1.append(max_val)
        scale_par_2.append(min_val)
        y_t[:,j] = (y_t[:,j]-scale_par_2[j])/(scale_par_1[j]-scale_par_2[j])
    else:
        mean_val = np.mean(y_t[:,j])
        std_val = np.std(y_t[:,j])
        scale_par_1.append(mean_val)
        scale_par_2.append(std_val)
        y_t[:,j] = (y_t[:,j]-scale_par_1[j])/scale_par_2[j]

np.save('scale_par_1.npy', scale_par_1)
np.save('scale_par_2.npy', scale_par_2)

x_train, x_, y_train, y_ = train_test_split(x_t,y_t,test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_,y_,test_size=0.50, random_state=1)
print("x_train.shape", x_train.shape, "y_train.shape", y_train.shape)
print("x_cv.shape", x_cv.shape, "y_cv.shape", y_cv.shape)
print("x_test.shape", x_test.shape, "y_test.shape", y_test.shape)

samples = x_train.shape[0]
features = x_train.shape[1]
print(samples, features)
#########################################################
class TestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_x, test_y):
        self.x = test_x
        self.y = test_y
        self.test_losses = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Evaluate on the test data
        test_loss = self.model.evaluate(self.x, self.y, verbose=0)
        self.test_losses.append(test_loss)

test_loss_callback = TestLossCallback(x_test, y_test)
#########################################################
model = Sequential([
   	tf.keras.Input(shape=(features,)), 
#    Lambda(add_noise),  # Add noise here
	Dense(units=2**8, activation="relu", name="L1"),
    Dense(units=2**7, activation="relu", name="L2"),
	Dense(units=2**6, activation="relu", name="L3"),
	Dense(units=4, activation="sigmoid", name="L4"),
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
history = model.fit(
   	x_train, y_train,
   	validation_data = (x_cv, y_cv),
   	epochs=500,
    callbacks=[test_loss_callback]
)

test_losses = test_loss_callback.test_losses

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

model.save('nn.keras')
