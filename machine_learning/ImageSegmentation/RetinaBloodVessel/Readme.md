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
<table >
    <tr>
        <th>Architecture</th>
        <th>Loss Function</th>
        <th>Learning Rate</th>
        <th>Epochs</th>
        <th>F1 Score</th>
        <th>Precision</th>
        <th>Recall</th>
    </tr>
    <tr>
        <th>UNet</th>
        <th>MSE</th>
        <th>0.001 </th>
        <th>100</th>
        <th>0.771</th>
        <th>0.810 </th>
        <th>0.7399</th>
    </tr>
    <tr>
        <th>UNet</th>
        <th>MSE</th>
        <th>0.001 </th>
        <th>175</th>
        <th>0.772 </th>
        <th>0.868 </th>
        <th>0.7043</th>
    </tr>
    <tr>
        <th>UNet</th>
        <th>MSE</th>
        <th>0.001 </th>
        <th>250</th>
        <th>0.7809 </th>
        <th>0.8029 </th>
        <th>0.7653</th>
    </tr>
    <tr>
        <th>UNet</th>
        <th>MSE</th>
        <th>0.001 </th>
        <th>300</th>
        <th>0.775 </th>
        <th>0.815 </th>
        <th>0.7454</th>
    </tr>
    <tr>
        <th>UNet</th>
        <th>L1</th>
        <th>0.001 </th>
        <th>100</th>
        <th>0.008 </th>
        <th>0.978 </th>
        <th>0.004</th>
    </tr>
    <tr>
        <th>UNet</th>
        <th>L1</th>
        <th>0.001 </th>
        <th>150</th>
        <th>0.016 </th>
        <th>0.638 </th>
        <th>0.008</th>
    </tr>
</table>

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