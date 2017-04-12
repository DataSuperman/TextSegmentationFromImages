# TextSegmentationFromImages
Takes input an image and detects the text inside the image and segments the image into block of text with XY-Cuts


detext.py --> this is the CNN that learns to identify text, so data is artificially created inside this class, since it requires a lot of time to train, I have just trained the network with handful of fonts and certain sizes. The network cannot recognize text bigger than a certain limit just because it is trained on small dataset. The network clearly identifies text with beginning and end of the text line. Network is trained for 4 hours and surprisingly performs good.

detext-heatmap.py --> This script creates a heat-map of pixels(text only) identified by the CNN. Just run the script things will be clearer. Problem takes 15 minutes to create heatmap of one image.

detext-train.py --> This script runs the network(detext.py).

Now the progress3 folder,
scaled.8.jpg --> This is one of the CV
heatmap.8.png --> This creates the heatmap of the above CV. Important thing to notice the heatmap doesn't take into account the picture, underlines etc. It is activated only on the text.
combo.png --> This is a side by side image of original CV and xy-cut performed on the heatmap generated by the CNN on that CV.

progress2 folder 
heatmap-2.png --> Side by side comparison of image with its heatmap

progress5 folder
splitter.py --> Tried a different method to segment the image, first the image is subdivided into the leafs of a tree along horizontal and vertical axis just like XY-cut. Then a graph is made out of this leafs, where nodes are  represented by the leafs of the tree and edges are similarity score which groups a bunch of those nodes together. It is bottom up approach where if nodes are merged together if they are similar. Doesn't work