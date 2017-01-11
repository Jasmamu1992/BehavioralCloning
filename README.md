# BehavioralCloning
___
**Model Selected**

> This is a computer vision therefore it is clear that convolutional network architecture should be used  

> Unlike the traffic sign project, behavioral cloning is a regression problem where we are trying yo map images to a continuous 
steering angle signal.

> Since we are interested only in the steering angle, the number of output of the network should be one. This is achieved by 
decreasing the dimentionality of output of convolutional layers with a series of fully connected layers

> After trying out few models I finally found that Nvidai model metioned in [Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) work better for behavioral cloning
___

**Model Architecture**

> The following model description is generated using Keras visualization tools

> Dropout is used with a keep probability of 0.5 in order to improve generalization

> Relu is used as activation function

![Model Architecture](https://github.com/Jasmamu1992/BehavioralCloning/blob/master/model.png)

___

**Training Data Augmentation**

> I started with the data provided by Udacity and visualized the distribution of steering angles

![Steering Angles Histogram](https://github.com/Jasmamu1992/BehavioralCloning/blob/master/Screenshot%20from%202017-01-10%2019-35-24.png)

*Since most of the steering anle values are almost zero and the number of taining data is small, additional data is generated using the following techniques

*Brightness*

![Brightness](https://github.com/Jasmamu1992/BehavioralCloning/blob/master/Brightness.png)

*Shift*

![Brightness](https://github.com/Jasmamu1992/BehavioralCloning/blob/master/Shift.png)

*Flip*

![Flip](https://github.com/Jasmamu1992/BehavioralCloning/blob/master/Flip.png)

*Shadows*

![Shadows](https://github.com/Jasmamu1992/BehavioralCloning/blob/master/Shadows.png)







