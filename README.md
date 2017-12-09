Introduction
============

This page contains software and instructions for [Gracker: A Graph-based Planar Object Tracker](1).  


Instructions
============

The package contains the following files and folders:

- "./data": This folder contains a sample video "nl_newspaper" from the TMT dataset (2). If you want to test other videos from TMT or UCSB (3), plese download and place them in this folder.

- "./save": This folder contains the tracking results by the proposed Gracker algorithm.

OpenCV 2.0 is neccesary for compiling and running the source code of this project.


Running
============

We also provide the compiled executable files in the "./demo" folder, which are independent from Visual Studio and OpenCV. 

Command format:
	GraphTracker [VideoName] [ShowMode];

Sample: 
	GraphTracker nl_newspaper 1;


References
==========

[1] T. Wang, and H. Ling, "Gracker: A Graph-based Planar Object Tracker", PAMI, 2017.

[2] A. Roy, X. Zhang, N. Wolleb, C. P. Quintero, and M. J?gersand. "Tracking
benchmark and evaluation for manipulation tasks", ICRA, 2015.

[3]  S. Gauglitz, T. H?llerer, and M. Turk. "Evaluation of interest point
detectors and feature descriptors for visual tracking", IJCV, 2011.



Copyright
=========

This software is free for use in research projects. If you
publish results obtained using this software, please cite our paper.
  

If you have any question, please feel free to contact Tao Wang(twang@bjtu.edu.cn).
