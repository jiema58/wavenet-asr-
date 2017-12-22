import tensorflow as tf
import numpy as np
import os
from collections import Counter
import librosa

class wavenet:
    def __init__(self,**param):
        self.n_blocks=param.get('residual_block_num',3)
        #self.batch_size=param.get('batch_size',10)
        self.word_size=param.get('word_size',2666)
        #self.data_path=param.get('data_path','D:/dataset/ChineseSpeechDatabase/train_mfcc.tfrecords')
        
    def conv1d(self,input_tensor,name,dim=128,size=1,stride=1,padding='SAME',dilation_rate=1,bias=False,causal=False):
        if causal:
            input_tensor=tf.pad(input_tensor,[[0,0],[dilation_rate*(size-1),0],[0,0]],'CONSTANT')
            padding='VALID'
        return tf.layers.conv1d(input_tensor,filters=dim,kernel_size=size,strides=stride,padding=padding,dilation_rate=dilation_rate,use_bias=bias,kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name) 
    
    def batch_norm(self,input_tensor,name,is_training):
        return tf.layers.batch_normalization(input_tensor,momentum=.99,training=is_training,name=name)
    
    def residual(self,input_tensor,dim,size,rate,name,is_training):
        with tf.variable_scope(name):
            out1=self.conv1d(input_tensor,'rc1',dim=dim,size=size,dilation_rate=rate)
            out1=self.batch_norm(out1,'bn1',is_training=is_training)
            out2=self.conv1d(input_tensor,'rc2',dim=dim,size=size,dilation_rate=rate)
            out2=self.batch_norm(out2,'bn2',is_training=is_training)
            out1=tf.nn.sigmoid(out1)
            out2=tf.nn.tanh(out2)
            out=out1*out2
            out=self.conv1d(out,'rc3',dim=dim)
            out=self.batch_norm(out,'bn3',is_training=is_training)
            out=tf.nn.tanh(out)
            output=input_tensor+out
            return output,out
        
    def run(self,x,is_training=True,reuse=False):
        with tf.variable_scope('model',reuse=reuse): 
            out=self.conv1d(x,'first_layer',dim=128,causal=True)
            out=self.batch_norm(out,'bn1',is_training=is_training)
            out=tf.nn.tanh(out)
            skip=0.
            with tf.variable_scope('residual_block'):
                for i in range(self.n_blocks):
                    with tf.variable_scope('stacked'+str(i+1)):
                        out,s1=self.residual(out,128,size=7,rate=1,name='rb_1',is_training=is_training)
                        out,s2=self.residual(out,128,size=7,rate=2,name='rb_2',is_training=is_training)
                        out,s3=self.residual(out,128,size=7,rate=4,name='rb_3',is_training=is_training)
                        out,s4=self.residual(out,128,size=7,rate=8,name='rb_4',is_training=is_training)
                        out,s5=self.residual(out,128,size=7,rate=16,name='rb_5',is_training=is_training)
                        skip=skip+s1+s2+s3+s4+s5
            out=self.conv1d(skip,name='final_layer1',dim=128)
            out=self.batch_norm(out,'bn2',is_training=is_training)
            out=tf.nn.tanh(out)
            final=self.conv1d(out,name='output_layer',dim=self.word_size,bias=True)
            return final


def train(filename,epoch=10,batch_size=10):
    lr=tf.placeholder(dtype=tf.float32,shape=[])
    wav=wavenet(batch_size=batch_size)
    filename_queue=tf.train.string_input_producer([filename],num_epochs=epoch)
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,features={'mfcc':tf.FixedLenFeature([40380,],tf.float32),'label':tf.FixedLenFeature([75,],tf.int64),})
    wave=features['mfcc']
    labels=tf.cast(features['label'],tf.int32)
    wave_batch,label_batch=tf.train.shuffle_batch([wave,labels],batch_size=batch_size,capacity=6*batch_size,min_after_dequeue=2*batch_size)
    wave_batch=tf.reshape(wave_batch,[batch_size,-1,60])
    logits=wav.run(wave_batch)
    sequence_len=tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(wave_batch, reduction_indices=2), 0.), tf.int32), reduction_indices=1)
    indices=tf.where(tf.not_equal(tf.cast(label_batch,tf.float32),0.))
    target=tf.SparseTensor(indices=indices,values=tf.gather_nd(label_batch,indices)-1,dense_shape=tf.cast(tf.shape(label_batch),tf.int64))
    ctc_loss=tf.nn.ctc_loss(target,logits,sequence_len,time_major=False)
    loss=tf.reduce_mean(ctc_loss)
    op=tf.train.AdamOptimizer(learning_rate=lr)
    var_list=[t for t in tf.trainable_variables()]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_op=op.minimize(loss) 
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            step=0
            while not coord.should_stop():
                step+=1
                curr_lr=0.00002*(0.97**(step//890))
                _,cost=sess.run([optimizer_op,loss],{lr:curr_lr})
                if step%100==0:
                    print('### Step: {}, Cost: {} ###'.format(step,cost))
                    saver.save(sess,'Wavenet_model/speech_recognition.ckpt',global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done for {} epochs, {} steps'.format(epoch,step))
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        finally:
            coord.request_stop()
        coord.join(threads)  
        #saver.save(os.path.join(os.getcwd(),'Wavenet_model/wavenet.ckpt'))