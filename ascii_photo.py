from PIL import Image,ImageDraw,ImageFont
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math

def f(x):
	return (math.sin(math.pi/255*x-math.pi/2)+1)/2

class Model:
	def __init__(self):
		self.chars=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
		self.func=np.vectorize(f)
		self.x=tf.placeholder(tf.float32,[None,48])
		self.y=tf.placeholder(tf.float32,[None, len(self.chars)])
		self.im=tf.reshape(self.x,[-1,8,6,1])
		self.c1=tf.layers.conv2d(self.im,5,2,1,'same',activation=tf.nn.relu)
		self.flat=tf.layers.flatten(self.c1)
		self.out=tf.layers.dense(self.flat,len(self.chars),tf.nn.sigmoid)
		self.losss=tf.losses.mean_squared_error(self.y,self.out)
		self.tra=tf.train.AdamOptimizer(0.001).minimize(self.losss)
		self.sess=tf.Session()
		self.saver = tf.train.Saver()
		#self.saver.restore(self.sess, './ascii')
		self.sess.run(tf.global_variables_initializer())
	def predict(self,tab):
		return self.sess.run(self.out,{self.x:[self.func(tab)]})
	def lern(self,tab,outt):
		_, lo=self.sess.run([self.tra,self.losss],{self.x:[self.func(tab)], self.y:[outt]})
		return lo
	def save(self):
		self.saver.save(self.sess, './ascii', write_meta_graph=False)

def images(model):
	tab=[]
	font=ImageFont.truetype('C:\\Users\\michalrz\\Desktop\\Programy-Gry\\Teemo-game\\font\\arial.ttf',6)
	for i in model.chars:
		temp=Image.new('L',(6,8),255)
		d=ImageDraw.Draw(temp)
		w, h = d.textsize(i,font=font)
		d.text(((6-w)/2,(8-h)/2),i,0,font)
		tab.append(temp)
	return tab

def presentation(model):
	n=Image.new('L',(1080,480),255)
	tab=images(model)
	for i in range(len(tab)):
		n.paste(tab[i],(i*6,0))
	n.show()
	
def black_white(name,model):
	tab=images(model)
	im=Image.open('C:\\Users\\michalrz\\Desktop\\jakieś_grafiki\\'+name)
	im=im.crop((0,0,im.size[0]-im.size[0]%6,im.size[1]-im.size[1]%8))
	im=im.convert('L')
	n=Image.new('L',im.size,255)
	for i in range(int(n.size[0]/6)):
		for j in range(int(n.size[1]/8)):
			p=np.argmax(model.predict(list(im.crop((i*6,j*8,i*6+6,j*8+8)).getdata())))
			n.paste(tab[p],(i*6,j*8))
	n.show()
	n.save('C:\\Users\\michalrz\\Desktop\\jakieś_grafiki\\ascii'+name,"JPEG")

def color(name,model):
	tab=images(model)
	font=ImageFont.truetype('C:\\Users\\michalrz\\Desktop\\Programy-Gry\\Teemo-game\\font\\arial.ttf',8)
	imm=Image.open('C:\\Users\\michalrz\\Desktop\\jakieś_grafiki\\'+name)
	imm=imm.crop((0,0,imm.size[0]-imm.size[0]%6,imm.size[1]-imm.size[1]%8))
	im=imm.convert('L')
	n=Image.new('RGB',im.size,255)
	for i in range(int(n.size[0]/6)):
		for j in range(int(n.size[1]/8)):
			p=np.argmax(model.predict(list(im.crop((i*6,j*8,i*6+6,j*8+8)).getdata())))
			
			col=[0,0,0]
			ii=list(imm.crop((i*6,j*8,i*6+6,j*8+8)).getdata())
			for k in ii:
				col[0]+=k[0]
				col[1]+=k[1]
				col[2]+=k[2]
			col[0]/=48
			col[1]/=48
			col[2]/=48
			
			temp=Image.new('RGB',(6,8),(255,255,255))
			d=ImageDraw.Draw(temp)
			w, h = d.textsize(model.chars[p],font=font)
			d.text(((6-w)/2,(8-h)/2),model.chars[p],(round(col[0]),round(col[1]),round(col[2])),font)
			
			n.paste(temp,(i*6,j*8))
	n.show()
	n.save('C:\\Users\\michalrz\\Desktop\\jakieś_grafiki\\ascii'+name,"JPEG")


def learn(model):
	tab=images(model)
	os.chdir('C:\\Users\\michalrz\\Desktop\\snakem\\ascii')
	loss=[]
	score=[]
	pop=0
	while pop<len(model.chars):
		pop=0
		
		for i in range(len(model.chars)):
			if np.argmax(model.predict(list(tab[i].getdata())))==i: pop+=1
			p=np.zeros(len(model.chars))
			p[i]=1
			loss.append(model.lern(list(tab[i].getdata()),p))
		score.append(pop)
		plt.clf()
		plt.subplot(2,1,1)
		plt.plot(range(1,len(loss)+1),loss)
		plt.subplot(2,1,2)
		plt.plot(range(1,len(score)+1),score)
		plt.pause(0.0001)
		if len(score)>=1000: break
	for _ in range(150):
		pop=0
		for i in range(len(model.chars)):
			if np.argmax(model.predict(list(tab[i].getdata())))==i: pop+=1
			p=np.zeros(len(model.chars))
			p[i]=1
			loss.append(model.lern(list(tab[i].getdata()),p))
		score.append(pop)
		plt.clf()
		plt.subplot(2,1,1)
		plt.plot(range(1,len(loss)+1),loss)
		plt.subplot(2,1,2)
		plt.plot(range(1,len(score)+1),score)
		plt.pause(0.0001)
	model.save()

os.chdir('C:\\Users\\michalrz\\Desktop\\snakem\\ascii')
model=Model()
presentation(model)
learn(model)
black_white('leonardo-da-vinci-mona-michał.jpg',model)
color('leonardo-da-vinci-mona-michał.jpg',model)