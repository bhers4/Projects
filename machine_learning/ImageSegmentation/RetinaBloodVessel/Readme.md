#  Retina Blood Vessel Segmentation using UNet
### Author: Ben Hers
### Architecture: UNet using 4 Encoder/Decoder blocks
#### Dataset: The Stare Project
Upon starting into the field of using deep learning for biomedical imaging, I started by working
on different object detection and image segmentation tasks and one of the most popular
architectures was UNet. I happened upon the STARE project from UCSD in different biomedical image
segmentation papers on Arxiv so I decided to use this dataset as a fun/interesting dataset to try
different network architectures for image segmentation and the effect of different loss functions.
An example of the dataset is shown below.
<br />
| Loss Function | Learning Rate | Epochs | F1 Score | Precision | Recall |
| MSE           | 0.001         | 100    | 0.771    | 0.810     | 0.7399 |

### Actual on left, Predicted on Right
<br />
<img src="outputImages/Org0.png" alt="drawing" width="300" height="300">
<br />
<p float="left">
<img src="outputImages/Actual0.png" alt="drawing" width="300" height="300">
<img src="outputImages/Pred0.png" alt="drawing" width="300" height="300">
</p>
<br />
<img src="outputImages/Org1.png" alt="drawing" width="300" height="300">
<br />
<p float="left">
<img src="outputImages/Actual1.png" alt="drawing" width="300" height="300">
<img src="outputImages/Pred1.png" alt="drawing" width="300" height="300">
</p>
<br />
<img src="outputImages/Org2.png" alt="drawing" width="300" height="300">
<br />
<p float="left">
<img src="outputImages/Actual2.png" alt="drawing" width="300" height="300">
<img src="outputImages/Pred2.png" alt="drawing" width="300" height="300">
</p>
<br />
<img src="outputImages/Org3.png" alt="drawing" width="300" height="300">
<br />
<p float="left">
<img src="outputImages/Actual3.png" alt="drawing" width="300" height="300">
<img src="outputImages/Pred3.png" alt="drawing" width="300" height="300">
</p>
<br />
<img src="outputImages/Org4.png" alt="drawing" width="300" height="300">
<br />
<p float="left">
<img src="outputImages/Actual4.png" alt="drawing" width="300" height="300">
<img src="outputImages/Pred4.png" alt="drawing" width="300" height="300">
</p>
<br />
<img src="lossperepoch1.png" alt="drawing" width="300" height="300">