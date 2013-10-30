
Please first see KSH_demo.m to find how my codes work. 

For a quick illustration, I used a sampled subset (9K) of the Photo Tourism (Notre Dame part) image patch 
dataset. Each image patch was represented by a 512-dimensional GIST feature vector. The raw database can 
be found from http://phototour.cs.washington.edu/patches/default.htm

One possible issue is kernel. I used Gaussian RBF kernel throughout my paper, but any other kernels are
admittable. Please ask me if you do not know how to incorporate other kernels. For the important parameter 
m used in my method, I just simply fix m=300.

For any problem with my codes, feel free to drop me a message via wliu@ee.columbia.edu. Also, I politely ask
you to cite my CVPR'12 paper in your publications.

Wei Liu
October 3, 2012

  






 