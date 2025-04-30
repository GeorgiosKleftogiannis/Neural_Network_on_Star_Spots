# Import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
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
###########################################################
for i in range(n_sample):
    m_x_t = np.mean(x_t[i,:])
    std_x_t = np.std(x_t[i,:])
    x_t[i,:] = (x_t[i,:] - m_x_t) / std_x_t

## for uniform distribution min-max scaling (min-max)
## for normal distribution Z scaling (mean-std)
scale_par_1 = [] # scale parameter 1 (mean or max_val)
scale_par_2 = [] # scale parameter 2 (std or min_val)
for j in range(4):
    if j <= 1:
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

#y_t = (y_t - min_val) / (max_val - min_val)
#y_t = (y_t - m_y_t) / std_y_t



########## scale train and target data ########################
#scaler = StandardScaler()
#x_t = scaler.fit_transform(x_t)
#y_t = scaler.fit_transform(y_t)
########## Create train, cross validate and test data ################
x_train, x_, y_train, y_ = train_test_split(x_t,y_t,test_size=0.40, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_,y_,test_size=0.50, random_state=1)
print("x_train.shape", x_train.shape, "y_train.shape", y_train.shape)
print("x_cv.shape", x_cv.shape, "y_cv.shape", y_cv.shape)
print("x_test.shape", x_test.shape, "y_test.shape", y_test.shape)
################################################
samples = x_train.shape[0]
features = x_train.shape[1]
print(samples, features)

#########  choose the best learning rate for Adam ########
a_list = [0.001, 0.01, 0.1, 1.0]
for a in a_list:
    model = Sequential([
   	    tf.keras.Input(shape=(features,)), 
	    Dense(units=512, activation="relu", name="L1"),
	    Dense(units=256, activation="relu", name="L2"),
	    Dense(units=4, name="L6"),
        ])
    print("learning rate = ",a)
    model.compile(optimizer=Adam(learning_rate=a), loss='mean_squared_error')
    history = model.fit(
   	    x_train, y_train,
   	    validation_data = (x_cv, y_cv),
   	    epochs=500
        )
    label_1 = 'Training Loss '+str(a)
    label_2 = 'Validation Loss '+str(a)
    plt.plot(history.history['loss'], label = label_1)
    plt.plot(history.history['val_loss'], label = label_2)

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

######## after running the above run for best learning rate #######
#train_loss = []
#cv_loss = []
#test_loss = []
#lamda = []
#for i in range(101):
#	lmd = i*0.001
#	model = Sequential([
#    tf.keras.Input(shape=(features,)), 
#	Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lmd), name="L1"),
#	Dense(units=32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lmd), name="L2"),
#	Dense(units=4, kernel_regularizer=tf.keras.regularizers.l2(lmd), name="L3"),
#			])
#	model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#	history = model.fit(
#   		x_train, y_train,
#   		validation_data = (x_cv, y_cv),
#   		epochs=500
#	)
#	########## calculate train and test error ########
#	train_loss.append(model.evaluate(x_train, y_train, verbose=0))
#	cv_loss.append(model.evaluate(x_cv, y_cv, verbose=0))
#	test_loss.append(model.evaluate(x_test, y_test, verbose=0))
#	lamda.append(lmd)

# print(f"training err {train_loss:0.2f}, cv err {cv_loss:0.2f}, test err {test_loss:0.2f}")
