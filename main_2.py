from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from inference import *
from rnn import *
import numpy as np
import skimage.measure


fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")
fout_pre= open("pred.txt","w")
fout_pres=open("pred_s.txt","w")
fout_gt=open("gt.txt","w")
fout_pre.write("\nPrediction \n"); fout_gt.write("\nGround Truth \n");
# Import Input data : Takes long time
train_size=250; test_size=50;
w=129;h=86; 
#Before start please run "python read_data.py" to save input file#
tr_input_image=np.load("/export/kim79/h2/finished_npy/5/tr_input_batch.npy")
tr_output_state=np.load("/export/kim79/h2/finished_npy/5/tr_output_state_batch.npy")
tr_output_lonlat=np.load("/export/kim79/h2/finished_npy/5/tr_output_lonlat_batch.npy")
te_input_image=np.load("/export/kim79/h2/finished_npy/5/te_input_batch.npy")
te_output_state=np.load("/export/kim79/h2/finished_npy/5/te_output_state_batch.npy")
te_output_lonlat=np.load("/export/kim79/h2/finished_npy/5/te_output_lonlat_batchh.npy")

tr_output_lonlat=np.expand_dims(tr_output_lonlat,axis=3)
te_output_lonlat=np.expand_dims(te_output_lonlat,axis=3)


print(np.shape(tr_input_image[0]))
print(np.shape(tr_output_state[0]))
print(np.shape(tr_output_lonlat[0]))
display_step=10;
testing_step=1;
# Training Parameters
learning_rate = 0.0005
training_steps =200000
input_size =timesteps=24; 
feature_size=w*h;
channels=2;

# Network Parameters
lstm_size=num_hidden = 100 # hidden layer num of features
num_classes = 1 # Length of output
number_of_layers=1; #Start from only one layer

# tf Graph input
X = tf.placeholder("float", [FLAGS.batch_size, timesteps, feature_size,channels])
Y_state = tf.placeholder("float", [FLAGS.batch_size, timesteps, num_classes])
Y_lonlat = tf.placeholder("float", [FLAGS.batch_size, timesteps, num_classes,2])
xx=tf.reshape(X,[FLAGS.batch_size*timesteps, h,w,2])
x_em=inference(xx);

print(x_em);
logits ,lonlat = RNN(x_em, weights, biases)

prediction_state=[]; prediction_lonlat=[];
for t in range(timesteps):
    prediction_state.append(tf.reshape(tf.nn.softmax(logits[t]),[FLAGS.batch_size,1,num_classes]));
    prediction_lonlat.append(tf.reshape(lonlat[t],[FLAGS.batch_size,1,num_classes,2]));
prediction_state = tf.concat(prediction_state, 1)
prediction_lonlat = tf.concat(prediction_lonlat, 1)

lon=prediction_lonlat[:,:,:,0]; lat=prediction_lonlat[:,:,:,1];
pred_lon=tf.reshape(tf.multiply(tf.reshape(lon,[FLAGS.batch_size*timesteps*num_classes]),tf.reshape(Y_state,[FLAGS.batch_size*timesteps*num_classes])),[FLAGS.batch_size,timesteps,num_classes,1])
pred_lat=tf.reshape(tf.multiply(tf.reshape(lat,[FLAGS.batch_size*timesteps*num_classes]),tf.reshape(Y_state,[FLAGS.batch_size*timesteps*num_classes])),[FLAGS.batch_size,timesteps,num_classes,1])
prediction_lonlat_temp=tf.concat([pred_lon,pred_lat],3);
print(tf.shape(Y_lonlat)); print(tf.shape(prediction_lonlat_temp));
prediction_lonlat=tf.concat([tf.reshape(Y_lonlat[:,0:1,:,:],[FLAGS.batch_size,1,num_classes,2]),prediction_lonlat_temp[:,1:timesteps,:,:]],1); #Always start from initial position
print(tf.shape(prediction_lonlat)); 



#check data type
fout_log.write(str(prediction_state)+"\n");
fout_log.write(str(prediction_lonlat)+"\n");
Y_sum=tf.reduce_sum(Y_state,2)
Y_sum=tf.reduce_sum(Y_sum,1)
Y_sum=tf.reduce_sum(Y_sum,0)
# Define loss and optimizer
loss_state = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_state, logits=prediction_state))
sqsum=tf.reduce_sum(tf.pow(prediction_lonlat - Y_lonlat,2))
loss_lonlat=tf.div(sqsum,Y_sum)
print(tf.shape(Y_sum));
print(tf.shape(loss_lonlat));

loss_op=loss_lonlat
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
        step_i=step%int(train_size);
        batch_x, batch_y_state, batch_y_lonlat = tr_input_image[step_i], tr_output_state[step_i], tr_output_lonlat[step_i];
       
        fout_log.write(str(step)+"\n");
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y_state: batch_y_state, Y_lonlat: batch_y_lonlat}) 
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, mse_tr = sess.run([loss_op, mse], feed_dict={X: batch_x, Y_state: batch_y_state, Y_lonlat: batch_y_lonlat})
            fout_log.write("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", MSE= " + \
                  "{:.3f}".format(mse_tr)+"\n")
        if (step % testing_step == 0):
            te_batch_x, te_batch_y_state, te_batch_y_lonlat = te_input_image[test_i], te_output_state[test_i], te_output_lonlat[test_i];
            test_i=test_i+1;
            test_mse=sess.run(mse, feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat}); 
            fout_log.write( "Testing MSE "+str(test_mse)+"\n")
            print( "Testing MSE "+str(test_mse)+"\n")
            #pre=sess.run(prediction_lonlat,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat});
            #pre_s=sess.run(prediction_state,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat}); 
            #gt=sess.run(Y_lonlat,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat});
            #for i in range(len(pre)):
            #    for j in range(len(pre[0])):
            #        fout_pre.write("\n"+str(step)+"th step "+str(i)+" ,"+str(j)+"\n");
            #        fout_gt.write("\n"+str(step)+"th step "+str(i)+" ,"+str(j)+"\n");
            #        for k in range(len(pre[0][0])):
            #            fout_pre.write(str(pre[i][j][k])+"\n");
            #            fout_pres.write(str(pre_s[i][j][k])+"\n");
            #            fout_gt.write(str(gt[i][j][k])+"\n");
        if (step % 100 == 0):
            #After finishing training Write Test result as output npy files
            print("DONE and writing test data")
            #Load test data set one
            pre_list=[]; gt_list=[];
            for stepp in range(len(te_output_state)):
                te_batch_x, te_batch_y_state, te_batch_y_lonlat = te_input_image[stepp], te_output_state[stepp], te_output_lonlat[stepp];
                pre=sess.run(prediction_lonlat,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat});
                gt=sess.run(Y_lonlat,feed_dict={X:te_batch_x, Y_state:te_batch_y_state, Y_lonlat: te_batch_y_lonlat});
                pre_list.append(pre);
                gt_list.append(gt);
                print("writing "+str(step)+"th testing result in set")
            pre_list=np.asarray(pre_list)
            gt_list=np.asarray(gt_list)
            np.save("prediction_"+str(step)+".npy",pre_list)
            np.save("ground_trunth_"+str(step)+".npy",gt_list)
fout_log.close();
fout_pre.close();
fout_gt.close();
