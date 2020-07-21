# Enhanced Image Prediction through Cahn-Hilliard Image Inpainiting
Pattern recognition, as a branch of the promising machine learning technique, has been applied in many fields. But insufficient information in the input may lead to unexpected results. In this work, I successfully enhanced the predicability of damaged binary images by applying image inpainting prior to a neural network model. This project combined the knowledge from classical chemical engineering field as well as the promising machine learning technique.

![image](https://github.com/fuyueliang/Enhanced-Image-prediction-through-Cahn-Hilliard-image-inpainiting-/blob/master/image/flow_chart.png)

To prove the enhancement, I used the numerical digits from MNIST handwritten database and created damage in advance. Subsequently, the accuracy of recognition before and after the restoration can be compared. In practice, we are unlikely to destroy the information in hand, the chief aim of this work is to give an idea that it is possible to extract useful information from a damaged image, if a neural network somehow successfully integrated with the inpainting process. 

## Functions description
The directory [*functions*](functions/) contains codes written in Python to execute the process described above. 

[Euler_implicit](functions/euler_implicit.py) defines the numerical solver which is subsequently applied to achieve image inpainting and [initial_conditions](functions/initial_conditions.py) defines the images and call the image inpainting solvers. These two files are imported to [image_inpainting](functions/image_inpainting.py) to complete the inpainting task on images from MNIST database. [Model_training](functions/model_training.py) and [prediction_model](functions/prediction_model.py) are executed prior to the numerical solver to build the neural network model. It is necessary to note that the model is setup only once at the beginning of the whole project, therefore this work is focused on the feasibility of improving images predictability by restoring the damaged domain as the identical prediciton model is used all the time.   

## Cahn-Hilliard equation solver
Cahn-Hilliard(CH) equation is originally used to model two-phase fluid dynamics, which is applied to complete image inpainting task in this project. A numerical scheme based on the finite volume method is constructed to solve the CH equation. In addition, a two-step method is employed to deal with the large damaged domains in images. Here are some examples of image inpainting.

![image](https://github.com/fuyueliang/Enhanced-Image-prediction-through-Cahn-Hilliard-image-inpainiting-/blob/master/image/example_inpainting.PNG)

## Neural network model
We defined our prediction model as a sequential model with one hidden layer, which is trained by introducing images from MNIST. I planned to apply deep learning at the very beginning, as the improvement of accuracy is satisfactory in the case of a shallow neural network, it is pointless to expand the size of hidden layer. When it comes to more complex recognition problems in the future, deep learning can be applicable. The following is the example of accuracy improvement.

![image](https://github.com/fuyueliang/Enhanced-Image-prediction-through-Cahn-Hilliard-image-inpainiting-/blob/master/image/accuracy_table.PNG)
