import tensorflow as tf
import numpy as np
import os
import librosa
from collections import Counter
from python_speech_features import mfcc
import scipy.io.wavfile

class Build_data:
    def __init__(self,**param):
        self.wave_file_path=param.get('wave_file','D:/dataset/ChineseSpeechDatabase/wav/train')
        self.label_file_path=param.get('label_file','D:/dataset/ChineseSpeechDatabase/doc/trans/train.word.txt')
        self.output_file=param.get('tfrecord','D:/dataset/ChineseSpeechDatabase/train_mfcc.tfrecords')
        
    def get_wav_files(self):
        wav_files=[]
        for (dirpath,dirnames,filenames) in os.walk(self.wave_file_path):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path=os.sep.join([dirpath,filename])
                    if os.stat(filename_path).st_size<240000:
                        continue
                    wav_files.append(filename_path)
        return wav_files

    def get_wav_label(self):
        wave_path=self.get_wav_files()
        labels_dict={}
        with open(self.label_file_path,'r',encoding='utf-8') as f:
            for label in f.readlines():
                label=label.strip('\n')
                label_id=label.split(' ',1)[0]
                label_text=label.split(' ',1)[1]
                labels_dict[label_id]=label_text
        self.labels=[]
        self.new_wave_path=[]
        for path in wave_path:
            wav_id=os.path.basename(path).split('.')[0]
            if wav_id in labels_dict:
                self.labels.append(labels_dict[wav_id])
                self.new_wave_path.append(path)
        print('yang ben shu',len(self.labels))
                
    def word_size(self):
        all_words=[]
        for label in self.labels:
            for v in label:
                all_words.append(v)
        dictionary=Counter(all_words)
        count_pairs=sorted(dictionary.items(),key=lambda x:-x[-1])
        word,_=zip(*count_pairs)
        self.word_size=len(word)
        print('total word number: ',self.word_size)
        self.word_to_num=dict(zip(word,range(self.word_size)))
        self.num_to_word=dict(zip(range(self.word_size),word))
    
    def record(self):
        self.get_wav_label()
        self.word_size()
        m=len(self.new_wave_path)
        writer=tf.python_io.TFRecordWriter(self.output_file)
        max_len=0
        for i in range(m):
            wav,sr=librosa.load(self.new_wave_path[i])
            mfcc=librosa.feature.mfcc(wav,sr,n_mfcc=20)
            mfcc_delta=librosa.feature.delta(mfcc)
            mfcc_delta_delta=librosa.feature.delta(mfcc,order=2)
            f_mfcc=np.vstack((mfcc,mfcc_delta,mfcc_delta_delta))
            f_mfcc=np.transpose(f_mfcc,[1,0])
            f_mfcc=np.reshape(f_mfcc,(-1))
            mfcc_s=np.zeros(40380)
            mfcc_s[0:len(f_mfcc)]=f_mfcc
            mfcc_s=list(mfcc_s)
            label=self.labels[i]
            #print(label)
            label_n=[self.word_to_num[w] for w in label]
            #print(label_n)
            label_n.extend([0]*(75-len(label_n)))
            e=self.example(mfcc_s,label_n)
            writer.write(e)
        writer.close()
    
    def example(self,mfcc,label):
        example=tf.train.Example(features=tf.train.Features(feature={'mfcc':tf.train.Feature(float_list=tf.train.FloatList(value=mfcc)),
                                                                     'label':tf.train.Feature(int64_list=tf.train.Int64List(value=label))}))
        return example.SerializeToString()
            



k=Build_data()
k.record()
