# Enhanced Image prediction through Cahn-Hilliard image inpainiting
Pattern recognition, as a branch of the promising machine learning technique, has been applied in many fields. But insufficient information in the input may lead to unexpected results. In this work, I successfully enhanced the predicability of damaged binary images by applying image inpainting prior to a neural network model. This project combined the knowledge from classical chemical engineering field as well as the promising machine learning technique.

![image](https://github.com/fuyueliang/Enhanced-Image-prediction-through-Cahn-Hilliard-image-inpainiting-/blob/master/images_pdf/flow_chart.png)

To prove the enhancement, I used the numerical digits from MNIST handwritten database and created damage in advance. Subsequently, the accuracy of recognition before and after the restoration can be compared. In practice, we are unlikely to destroy the information in hand, the chief aim of this work is to give an idea that it is possible to extract useful information from a damaged image, if a neural network somehow successfully integrated with the inpainting process.   

## Cahn-Hilliard Equation Solver
Cahn-Hilliard(CH) equation is originally used to model two-phase fluid dynamics, which is applied to complete image inpainting task in this project. A numerical scheme based on the finite volume method is constructed to solve the CH equation. In addition, a two-step method is employed to deal with the large damaged domains in images. Here are some examples of image inpainting.

![image](https://github.com/fuyueliang/Enhanced-Image-prediction-through-Cahn-Hilliard-image-inpainiting-/blob/master/images_pdf/example_inpainting.PNG)

## Neural network model
We defined our prediction model as a sequential model with one hidden layer, which is trained by introducing images from MNIST. I planned to apply deep learning at the very beginning, as the improvement of accuracy is satisfactory in the case of a shallow neural network, it is pointless to expand the size of hidden layer. When it comes to more complex recognition problems in the future, deep learning can be applicable. The following is the example of accuracy improvement.

![image](https://github.com/fuyueliang/Enhanced-Image-prediction-through-Cahn-Hilliard-image-inpainiting-/blob/master/images_pdf/accuracy_table.PNG)
