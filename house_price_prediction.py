import tensorflow as tf
import numpy as np # random number
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

# house size generator
num_house = 160
np.random.seed(10)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

#house price generator
np.random.seed(10)
house_price = house_size * 100.0 + np.random.randint(low = 10000, high = 50000, size=num_house)

#plot
plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
#plt.show()

def normalize(array): 
    return (array - array.mean()) / array.std()

# 70% of training data
num_train_samples = int(math.floor(num_house * 0.7))

#training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

#test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

#http://bcho.tistory.com/1150
tf_house_size = tf.placeholder(dtype = tf.float32, name = "house_size")
tf_price = tf.placeholder(dtype = tf.float32, name = "price")

tf_size_factor = tf.Variable(np.random.randn(), name = "size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name = "price_offset")

#y=wx+b
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)

learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

#http://effbot.org/zone/python-with-statement.htm

#launch the graph
with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    num_training_iter = 70

    for i in range(num_training_iter):
        #https://www.saltycrane.com/blog/2008/04/how-to-use-pythons-enumerate-and-zip-to/
        for (x,y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size:x, tf_price:y})
        if( i + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
            print("iteration #:", '%04d'%(i+1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
            
    print("Optimization Done")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_house_price.mean()
    train_price_std = train_house_price.std()

    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()




