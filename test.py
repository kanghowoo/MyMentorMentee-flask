from data import *
from textcnn import TextCNN
from flask import Flask,request
import tensorflow as tf
import numpy as np
import boto3
 

TRAIN_FILENAME = 'ratings_train.txt'
TRAIN_DATA_FILENAME = TRAIN_FILENAME + '.data'
TRAIN_VOCAB_FILENAME = TRAIN_FILENAME + '.vocab'

SEQUENCE_LENGTH = 60
NUM_CLASS = 2

dynamodb = boto3.resource('dynamodb')
MyMentorDB = dynamodb.Table('MyMentorDB')

print('MyMentorDB table created time : ',MyMentorDB.creation_date_time)


app = Flask(__name__)
@app.route('/',methods=['POST'])
def test():

    with tf.Session() as sess:
        
        vocab = load_vocab(TRAIN_VOCAB_FILENAME)
        cnn = TextCNN(SEQUENCE_LENGTH, NUM_CLASS, len(vocab), 128, [3,4,5], 128)
        saver = tf.train.Saver()
        saver.restore(sess, './textcnn.ckpt')
        print('model restored')
		
        # http 통신 post 로 body 에 'str' 
        input_text = request.form['str']
        masterName = request.form['masterName']
		
        tokens = tokenize(input_text)
        print('입력 문장을 다음의 토큰으로 분해:')
        print(tokens)

        sequence = [get_token_id(t, vocab) for t in tokens]
        x = []
        while len(sequence) > 0:
            seq_seg = sequence[:SEQUENCE_LENGTH]
            sequence = sequence[SEQUENCE_LENGTH:]

            padding = [1] *(SEQUENCE_LENGTH - len(seq_seg))
            seq_seg = seq_seg + padding

            x.append(seq_seg)
        
        feed_dict = {
            cnn.input : x,
            cnn.dropout_keep_prob : 1.0
        }

        predict = sess.run([cnn.predictions], feed_dict)
        

        result = np.mean(predict) 
        if (result > 0.75):
            print('추천')
        elif (result < 0.25):
            print('비추천')
        else:
            print('평가 불가능')

    
    MyMentorDB.update_item(
        Key={
            'Username' : masterName
        },
        UpdateExpression='ADD grade :val',
        ExpressionAttributeValues = {
        ':val' : int(result)
        }
    )

    
    tf.reset_default_graph()
    
    return (str(result))
@app.errorhandler(500)
def error_found(error) :
    tf.reset_default_graph()
    return 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    #test()