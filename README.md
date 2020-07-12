# Enhanced Image prediction through Cahn-Hilliard image inpainiting
Pattern recognition, as a branch of the promising machine learning technique, can be applied in many field. But insufficient information in the input may lead to unexpected results. In this work, I successfully enhanced the predicability of damged images by applying image inpainting prior to a neural network model.  This project combined the knowledge from classical chemical engineering field as well as the promising machine learning technique.

## Cahn-Hilliard Equation Solver
Cahn-Hilliard(CH) equation is originally used to model two-phase fluid dynamics, which is applied to complete image inpainting task in this project. A numerical scheme based on the finite volume method is constructed to solve the CH equation. In addition, a two-step method is employed to deal with the large damaged domains in images.

## Image inpainting prior to prediction/recognition
Image inpaintning is a type of automated interpolation of damaged areas of an image based on the information collected from background environment. When it comes to damaged images, their predictability can be decreased depending on how much information were missed, namely the extent of damage. Instead of directly inputting the damaged images into prediction model, we applied image restoration prior to the recognition process. Subsequently, it is proved that the predicability is significantly improved as the image inpainting is involved.

##
