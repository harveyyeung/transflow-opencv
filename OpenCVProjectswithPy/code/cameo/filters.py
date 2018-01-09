import cv2
import numpy
import utils

# Simulating RC color space
def recolorRC(src, dst):
    """
    Simulate conversion from BGR to RC (red, cyan).
    The source and destination images must both be in BGR format.
    Blues and greens are replaced with cyans.
    Pseudocode:
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    # 使用split（），我们将源图像的通道提取为一维数组。把这些数据放在这个格式里，我们可以写出清晰，简单的通道混合码。
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    # 使用addWeighted（），
    # 我们用B和G的平均值代替B通道的值。
    # addWeighted（）的参数是（依次）
    # 第一个源数组，
    # 第一个数组元素的权重
    # 第二个源数组，
    # 应用于第二个源数组的权重，
    # 添加到结果的常量以及目标数组
    cv2.merge((b, b, r), dst)
    # 使用merge（），我们用修改的通道替换目标图像中的值。
    # 请注意，我们使用b两次作为参数，因为我们希望目标的B和G通道相等


def recolorRGV(src, dst):
    """Simulate conversion from BGR to RGV (red, green, value).
    The source and destination images must both be in BGR format.
    Blues are desaturated.
    Pseudocode:
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    # min()函数计算前两个参数的每个元素的最小值，
    # 并将它们写入第三个参数。
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    """Simulate conversion from BGR to CMV (cyan, magenta, value).
    The source and destination images must both be in BGR format.
    Yellows are desaturated.
    Pseudocode:
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    # max()函数计算前两个参数的每个元素的最大值，
    # 并将它们写入第三个参数。
    cv2.merge((b, g, r), dst)    


def strokeEdges(src, dst , blurKsize=7,edgeKsize = 5):
    if blurKsize >=3:
        blurredSrc = cv2.medianBlur(src,blurKsize)  
        graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc,cv2.CV_8U,graySrc,ksize=edgeKsize) 
    normalizedInverseAlpha = (1.0/255)* (255 - graySrc) 
    channels = cv2.split(src)
    for channel in channels:
        channel[:]= channel * normalizedInverseAlpha
        #将其归一化（使其像素值在0-1之间），并乘以源图像以便能将边缘变黑
    cv2.merge(channels,dst)   

class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)."""
    
    def __init__(self, vFunc = None, dtype = numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)
    
    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        srcFlatView = utils.flatView(src)
        dstFlatView = utils.flatView(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView,
                               dstFlatView)

class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""
    
    def __init__(self, vPoints, dtype = numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints),
                             dtype)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""
    
    def __init__(self, vFunc = None, bFunc = None, gFunc = None,
                 rFunc = None, dtype = numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), length)
    
    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""
    
    def __init__(self, vPoints = None, bPoints = None,
                 gPoints = None, rPoints = None, dtype = numpy.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)

class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""
    
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints = [(0,20),(255,235)],
            gPoints = [(0,0),(56,39),(208,226),(255,255)],
            rPoints = [(0,0),(56,22),(211,255),(255,255)],
            dtype = dtype)

class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""
    
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints = [(0,0),(23,20),(157,173),(255,255)],
            bPoints = [(0,0),(41,46),(231,228),(255,255)],
            gPoints = [(0,0),(52,47),(189,196),(255,255)],
            rPoints = [(0,0),(69,69),(213,218),(255,255)],
            dtype = dtype)

class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""
    
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints = [(0,0),(35,25),(205,227),(255,255)],
            gPoints = [(0,0),(27,21),(196,207),(255,255)],
            rPoints = [(0,0),(59,54),(202,210),(255,255)],
            dtype = dtype)

class BGRVelviaCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""
    
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints = [(0,0),(128,118),(221,215),(255,255)],
            bPoints = [(0,0),(25,21),(122,153),(165,206),(255,255)],
            gPoints = [(0,0),(25,21),(95,102),(181,208),(255,255)],
            rPoints = [(0,0),(41,28),(183,209),(255,255)],
            dtype = dtype)        

class VConvolutionFilter(object):
     """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self,kernel):
        self._kernel= kernel
    
    def apply(self,src ,dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src,-1,self._kernel,dst)

class SharpenFilter(VConvolutionFilter):
      """A sharpen filter with a 1-pixel radius."""
        def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)      

'''注意，只要我们想保持图像的整体亮度不变,权重总和就要1.
如果我们稍微修改一个锐化内核，使它​​的权重总和为0，
那么我们有一个边缘检测内核，它将边缘变成白色，而非边缘变成黑色。
例如，让我们将下面的边缘检测过滤器FindEdgesFilter添加到filters.py：'''
class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)        

'''
下面构建一个模糊滤波器。为了达到模糊的效果，通常权重和应该为1，
而且邻近像素的权重全为正。以下就是一个简单的邻近平均滤波器
'''
class BlurFilter(VConvolutionFilter):
"""A blur filter with a 2-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)

'''
我们的锐化，边缘检测和模糊滤镜使用高度对称的内核。
有时候，对称性较差的核心会产生一个有趣的效果。
如下介绍的核，同时具有模糊（有正的权重）和锐化（有负的权重）作用。
它会产生起伏或浮雕的效果。

'''        
class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                                [-1, 1, 1],
                                [ 0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)