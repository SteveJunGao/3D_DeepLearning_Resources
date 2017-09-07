# 3D Deep Learning Code Resources
 ![update](https://img.shields.io/badge/LastUpdate-09--06--2017-brightgreen.svg)

In this repository we present open source codes for 3D Deep Learning available online. The purpose of this project is to make it convenient for anyone interested in deep learning of 3D models or scenes to find the corresponding code of a particular paper with a single search. Currently this page is maintained by [heiwang1997](https://github.com/heiwang1997), [SteveJunGao](https://github.com/SteveJunGao) and [Fangyin Wei](). All contributions are welcome.

Following the tutorial from Hao in Stanford University, we classify related works into the three categories: *3D Geometry Analysis, 3D-assisted Image Analysis and 3D Synthesis*. Some papers are cross-category and if you find it difficult to browse by category, you can simply search within this file.

Note only papers with code releases will be covered in this repo and we do not guarantee the results in the original paper can be fully reproduced by the codes below.

## 3D Geometry Analysis

> This part will contain 3D classification, parsing, correspondence and segmentation.

Wu, Zhirong, et al. "**3d shapenets: A deep representation for volumetric shapes**." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2015. [[paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf) [[matlab]](https://github.com/zhirongw/3DShapeNets)

Su, Hang, et al. "**Multi-view convolutional neural networks for 3d shape recognition**." *Proceedings of the IEEE international conference on computer vision*. 2015. [[paper]](https://arxiv.org/pdf/1505.00880) [[matlab]](https://github.com/suhangpro/mvcnn) [[caffe]](https://github.com/suhangpro/mvcnn/tree/master/caffe) [[tensorflow]](https://github.com/WeiTang114/MVCNN-TensorFlow) [[torch]](https://github.com/eriche2016/mvcnn.torch)

Li, Yangyan, et al. "**Joint embeddings of shapes and images via CNN image purification**." *ACM Trans. Graph.* 34.6 (2015): 234-1. [[paper]](https://shapenet.cs.stanford.edu/projects/JointEmbedding/JointEmbedding.pdf) [[caffe]](https://github.com/ShapeNet/JointEmbedding)

Su, Hao, et al. "**Render for cnn: Viewpoint estimation in images using cnns trained with rendered 3d model views**." *Proceedings of the IEEE International Conference on Computer Vision*. 2015. [[paper]](http://arxiv.org/abs/1505.05641) [[caffe]](https://github.com/shapenet/RenderForCNN/) 

Li, Yangyan, et al. "**Fpnn: Field probing neural networks for 3d data**." *Advances in Neural Information Processing Systems*. 2016. [[paper]](https://arxiv.org/pdf/1605.06240.pdf) [[caffe]](https://github.com/yangyanli/FPNN)

Maturana, Daniel, and Sebastian Scherer. "**Voxnet: A 3d convolutional neural network for real-time object recognition.**" *Intelligent Robots and Systems (IROS), 2015 IEEE/RSJ International Conference on*. IEEE, 2015. [[paper]](http://www.dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf) [[theano]](https://github.com/dimatura/voxnet) 

Qi, Charles R., et al. "**Volumetric and multi-view cnns for object classification on 3d data**." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016. [[paper]](https://arxiv.org/pdf/1604.03265) [[torch]](https://github.com/charlesq34/3dcnn.torch)

Kalogerakis, Evangelos, et al. "**3D Shape Segmentation with Projective Convolutional Networks**." *arXiv preprint arXiv:1612.02808* (2016). [[paper]](https://arxiv.org/pdf/1612.02808) [[project (code comming soon)]](http://people.cs.umass.edu/~kalo/papers/shapepfcn/)

Shi, Jian, et al. "**Learning Non-Lambertian Object Intrinsics across ShapeNet Categories**." *arXiv preprint arXiv:1612.08510* (2016). [[paper]](https://arxiv.org/pdf/1612.08510.pdf) [[torch]](https://github.com/shi-jian/shapenet-intrinsics)

Zeng, Andy, et al. "**3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions.**" *arXiv preprint arXiv:1603.08182* (2016). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf) [[project]](http://3dmatch.cs.princeton.edu/) [[matlab]](https://github.com/andyzeng/3dmatch-toolbox)

Pavlakos, Georgios, et al. "**Coarse-to-fine volumetric prediction for single-image 3D human pose**." *arXiv preprint arXiv:1611.07828* (2016). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Pavlakos_Coarse-To-Fine_Volumetric_Prediction_CVPR_2017_paper.pdf) [[project]](https://www.seas.upenn.edu/~pavlakos/projects/volumetric/) [[torch]](https://github.com/geopavlakos/c2f-vol-train/)

Xu, Kai, et al. "**3d attention-driven depth acquisition for object identification**." *ACM Transactions on Graphics (TOG)* 35.6 (2016): 238. [[paper]](http://kevinkaixu.net/papers/xu_siga16_nbv.pdf) [[torch]](https://github.com/kevin-kaixu/multi_view_ram)

Qi, Charles R., et al. "**Pointnet: Deep learning on point sets for 3d classification and segmentation**." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. [[paper]](https://arxiv.org/pdf/1612.00593.pdf) [[tensorflow]](https://github.com/charlesq34/pointnet) [[pytorch]](https://github.com/fxia22/pointnet.pytorch)

Tulsiani, Shubham, et al. "**Learning shape abstractions by assembling volumetric primitives**." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. [[paper]](https://arxiv.org/abs/1612.00404) [[torch]](https://github.com/shubhtuls/volumetricPrimitives)

Yi, Li, et al. "**Learning Hierarchical Shape Segmentation and Labeling from Online Repositories.**" *arXiv preprint arXiv:1705.01661* (2017). [[paper]](https://arxiv.org/abs/1705.01661) [[code (project page)]](http://cs.stanford.edu/~ericyi/project_page/hier_seg/index.html)

Yi, Li, et al. "**Syncspeccnn: Synchronized spectral CNN for 3d shape segmentation**." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. [[paper]](https://arxiv.org/pdf/1612.00606.pdf) [[torch]](https://github.com/ericyi/SyncSpecCNN)

Tome, Denis, Chris Russell, and Lourdes Agapito. "**Lifting from the deep: Convolutional 3d pose estimation from a single image**." *arXiv preprint arXiv:1701.00295* (2017). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tome_Lifting_From_the_CVPR_2017_paper.pdf) [[tensorflow]](https://github.com/DenisTome/Lifting-from-the-Deep-release)

Lin, Mude, et al. "**Recurrent 3D Pose Sequence Machines.**" *arXiv preprint arXiv:1707.09695* (2017). [[paper]](https://arxiv.org/abs/1707.09695) [[torch]](https://github.com/Geekking/RPSM)

Riegler, G., A. O. Ulusoy, and A. Geiger. "**OctNet: Learning deep 3D representations at high resolutions.**" *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. [[paper]](https://arxiv.org/pdf/1611.05009.pdf) [[c/cuda]](https://github.com/griegler/octnet) 

Wang, Peng-Shuai, et al. "**O-CNN: octree-based convolutional neural networks for 3D shape analysis**." *ACM Transactions on Graphics (TOG)* 36.4 (2017): 72. [[paper]](http://wang-ps.github.io/O-CNN_files/CNN3D.pdf) [[caffe]](https://github.com/Microsoft/O-CNN)

Deng, Zhuo, and Longin Jan Latecki. "**Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images**." [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Deng_Amodal_Detection_of_CVPR_2017_paper.pdf) [[caffe]](https://github.com/phoenixnn/Amodal3Det)

Dai, Angela, et al. "**Scannet: Richly-annotated 3d reconstructions of indoor scenes**." *arXiv preprint arXiv:1702.04405* (2017). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf) [[project]](http://www.scan-net.org/) [[github]](https://github.com/ScanNet/ScanNet)

## 3D-assisted Image Analysis
>  This part will contain cross-view image retrieval, intrinsic decomposition.

Su, Hao, et al. "**3D-assisted feature synthesis for novel views of an object.**" *Proceedings of the IEEE International Conference on Computer Vision*. 2015. [[paper]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_3D-Assisted_Feature_Synthesis_ICCV_2015_paper.pdf) 

## 3D Synthesis

> This part will contain monocular 3D reconstruction, shape completion and shape modeling.

Fan, Haoqiang, Hao Su, and Leonidas Guibas. "**A point set generation network for 3d object reconstruction from a single image**." *arXiv preprint arXiv:1612.00603* (2016). [[paper]](https://arxiv.org/abs/1612.00603) [[tensorflow]](https://github.com/fanhqme/PointSetGeneration)

Tran, Anh Tuan, et al. **"Regressing robust and discriminative 3D morphable models with a very deep neural network**." *arXiv preprint arXiv:1612.04904*(2016). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tran_Regressing_Robust_and_CVPR_2017_paper.pdf) [[project]](https://github.com/anhttran/3dmm_cnn) [[pycaffe]](http://www.openu.ac.il/home/hassner/projects/CNN3DMM/)

Park, Eunbyung, et al. "**Transformation-grounded image generation network for novel 3d view synthesis**." *arXiv preprint arXiv:1703.02921* (2017). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Park_Transformation-Grounded_Image_Generation_CVPR_2017_paper.pdf) [[project]](http://www.cs.unc.edu/~eunbyung/tvsn/)

Sinha, Ayan, et al. "**SurfNet: Generating 3D shape surfaces using deep residual networks**." *arXiv preprint arXiv:1703.04079* (2017). [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sinha_SurfNet_Generating_3D_CVPR_2017_paper.pdf) [[matlab]](https://github.com/sinhayan/surfnet)
