# Semantic Segmentation

Implemented Semantic Segmentation using ResNet18's structure and experimented with overfitted model's and regularly trained model's performance using miou as the evaluation metric.

The project uses Pascal dataset http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html , which provides standardised image data sets for object class recognition with 21 classes (20 Classes + 1 "other"/"background"). In our case, we utilized ResNet18's first 5 layers' structure with skip connections added to perform visual object classification.

Each class is represented in pixels with different colors. Note that in our case, we are performing semantic segmentation. Instead of recognizing an object from an image, we want more. We want to predict which class does each pixel belong to as well. From the below, we can see that black pixels represent background. The 20 classes are: distribution among Person(person), Animal (bird, cat, cow, dog, horse, sheep), Vehicle (aeroplane, bicycle, boat, bus, car, motorbike, train) and Indoor (bottle, chair, dining table, potted plant, sofa, tv/monitor).

Below are some examples about the dataset we are using. 

<img width="980" alt="Screen Shot 2022-01-01 at 6 51 29 PM" src="imgs/sampleImg.png">

The metric for model evaluation is mean Intersection over Union (mIoU), which focuses only on segment/classes, which is irrelevant to object sizes. mIoU is defined as below (screenshot from chainerCV documentation page:https://chainercv.readthedocs.io/en/stable/reference/evaluations.html#semantic-segmentation-iou ):

Nij in the below formula represents the number of pixels that are labeled as class i by the ground truth and class j by the prediction.

![Screen Shot 2022-01-01 at 7 00 17 PM](https://user-images.githubusercontent.com/54965707/147862685-79a6e50d-ad3a-4139-b30e-7e78286e3264.png)

For each class, we calculate how similar is our prediction to the actual object pixelwise. More mathematically, we first calculate the intersection between our prediction and groud truth, and then we calculate the union of the prediction and ground truth. We then take the ratio of those two numbers. We have the ratio number for each class. Then, we average over all k classes, which is 21 in our case.


In our model, we used ResNet18's first 5 blocks as our feature encoder to extract the features and then we use added skip connection to concatenate the previous feature map (skipped feature map) and the current feature map (upsampled feature map) as our new feature map and then learn the corresponding decoder filter to interpolate the image to the original shape. The concatenated feature map's corresponding convolutional layer is quite symmetirc to how we extracted our features into deep feature maps, which will help the model learn how to combine individual features.

During the model fitting, we performed image segmentation with RandomResizedCrop and Random Horizontal Flip as demonstrated below:

<p align="center">
  <img src="imgs/RandomHorizontalFlip.png" height = "400" width="480">
  <img src="imgs/RandomResizeCrop.png" height = "400"  width="480">
</p>

After experimenting and augmenting the data and its transformations, we evaluate our untrained model immediately and see its performance. In our case, we only achieved 0.06 mIoU.

<img src="results/untrained_performance.png" height = "250" width = "980"/>

Then we tried to overfit the model on one single image for 100 epoch. The loss graph is presented below.
<img src="results/overfit_training.png" height = "400" width = "980"/>

We can see that after 100 epoch, which is seriously overfitted, the loss is reduced / nearly converged. it reaches mIoU = accuracy on the trained image.

<img src="results/overfit_prediction1.png" height = "250" width = "980"/>

However, we can see that it performs poorly as expected on the new image. We can also see that our model simply tries to memorize the training image. When given a new image, it tries to classify based on the knowledge of memorizing one single image. We can see that the pixel class corresponding color is grey for the validation image and the pixel class corresponding color is gree for the training image. After overfitting, model tries to predict green color instead of grey.

<img src="results/overfit_prediction2.png" height = "250" width = "980"/>

Then, we started to train our model on the entire training set for 4 epochs with batch size = 4. The below is corresponding loss graph.

<img src="results/fit_training.png" height = "400" width = "980"/>

We also tried to validate our result. The accuracy is improved for test image and decreased a little bit for the training image, which is what we want to avoid overfitting.

<img src="results/fit_prediction1.png" height = "250" width = "980"/>

<img src="results/fit_prediction2.png" height = "250" width = "980"/>



