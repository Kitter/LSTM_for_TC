from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from read_data import *
from inference import *
from rnn import *
import numpy as np

fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")
fout_pre= open("pred.txt","w")
fout_pres=open("pred_s.txt","w")
fout_gt=open("gt.txt","w")
fout_pre.write("\nPrediction \n"); fout_gt.write("\nGround Truth \n");
# Import Input data : Takes long time
train_size=6500; test_size=500;
#train_size=8; test_size=2;

w=60;h=30;
tr_input_image, tr_output_state, tr_output_lonlat, te_input_image, te_output_state, te_output_lonlat=read_input(FLAGS.batch_size, train_size, test_size);


display_step=1;
testing_step=100;
# Training Parameters
learning_rate = 0.001
training_steps = 100000
input_size =timesteps=24; 
feature_size=w*h;

# Network Parameters
lstm_size=num_hidden = 100 # hidden layer num of features
num_classes = 13 # Length of output
number_of_layers=1; #Start from only one layer

# tf Graph input
X = tf.placeholder("float", [FLAGS.batch_size, timesteps, feature_size])
Y_state = tf.placeholder("float", [FLAGS.batch_size, timesteps, num_classes])
Y_lonlat = tf.placeholder("float", [FLAGS.batch_size, timesteps, num_classes,2])
channels=1;
xx=tf.reshape(X,[FLAGS.batch_size*timesteps, w,h,channels]);
x_em=inference(xx);
#x_em=old_embedding_1layer(xx);
print(x_em);

logits ,lonlat = RNN(x_em, weights, biases)
#logits ,lonlat = RNN(X, weights, biases)

prediction_state=[]; prediction_lonlat=[];
for t in range(timesteps):
    prediction_state.append(tf.reshape(tf.nn.softmax(logits[t]),[FLAGS.batch_size,1,num_classes]));
    prediction_lonlat.append(tf.reshape(lonlat[t],[FLAGS.batch_size,1,num_classes,2]));
print(lonlat[0])
prediction_state = tf.concat(prediction_state, 1)
prediction_lonlat = tf.concat(prediction_lonlat, 1)

lon=prediction_lonlat[:,:,:,0]; lat=prediction_lonlat[:,:,:,1];
pred_lon=tf.reshape(tf.multiply(tf.reshape(lon,[FLAGS.batch_size*timesteps*num_classes]),tf.reshape(Y_state,[FLAGS.batch_size*timesteps*num_classes])),[FLAGS.batch_size,timesteps,num_classes,1])
pred_lat=tf.reshape(tf.multiply(tf.reshape(lat,[FLAGS.batch_size*timesteps*num_classes]),tf.reshape(Y_state,[FLAGS.batch_size*timesteps*num_classes])),[FLAGS.batch_size,timesteps,num_classes,1])
prediction_lonlat=tf.concat([pred_lon,pred_lat],3);


#check data type
fout_log.write(str(prediction_state)+"\n");
fout_log.write(str(prediction_lonlat)+"\n");
# Define loss and optimizer
loss_state = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_state, logits=prediction_state))
loss_lonlat=tf.reduce_mean(tf.losses.mean_squared_error(labels=Y_lonlat,predictions=prediction_lonlat))
loss_op=loss_lonlat+loss_state
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#Evaluate Model
mse=loss_lonlat;

#Initialize the variables
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in xrange(1, training_steps+1):
        print(step); test_i=0;
        step=step%int(train_size);
        batch_x, batch_y_state, batch_y_lonlat = tr_input_image[step], tr_output_state[step], tr_output_lonlat[step];
        fout_log.write(str(step)+"\n");
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y_state: batch_y_state, Y_lonlat: batch_y_lonlat}) 
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, mse_tr = sess.run([loss_op, mse], feed_dict={X: batch_x, Y_state: batch_y_state, Y_lonlat: batch_y_lonlat})
            fout_log.write("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", MSE= " + \
                  "{:.3f}".format(mse_tr)+"\n")
        if step!=0 and (step % testing_step == 0):
            te_batch_x, te_batch_y_state, te_batch_y_lonlat = te_input_image[test_i], te_output_state[test_i], te_output_lonlat[test_i];
            test_i=test_i+1;
            test_mse=sess.run(mse, feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat}); 
            fout_log.write( "Testing MSE "+str(test_mse)+"\n")
            pre=sess.run(prediction_lonlat,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat});
            pre_s=sess.run(prediction_state,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat}); 
            gt=sess.run(Y_lonlat,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat}); #(8, 56, 13, 2)
            for i in range(len(pre)):
                for j in range(len(pre[0])):
                    fout_pre.write("\n"+str(step)+"th step "+str(i)+" ,"+str(j)+"\n");
                    fout_gt.write("\n"+str(step)+"th step "+str(i)+" ,"+str(j)+"\n");
                    for k in range(len(pre[0][0])):
                        fout_pre.write(str(pre[i][j][k])+"\n");
                        fout_pres.write(str(pre_s[i][j][k])+"\n");
                        fout_gt.write(str(gt[i][j][k])+"\n");
fout_log.close();
fout_pre.close();
fout_gt.close();
