class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, target):
        assert img.size == target.size
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
class JointToTensor(object):
    def __call__(self, img, target):
        return tF.to_tensor(img), torch.from_numpy(np.array(target.convert('P'), dtype=np.int32)).long()
    
class JointRandomHorizontalFlip(object):
    def __init__(self):
        return
    def __call__(self, img, target):
        rand = random.randint(0,1)
        if rand == 0:
            return img, target
        else:
            return tF.hflip(img), tF.hflip(target)
class JointCenterCrop(object):
    def __init__(self, size):
        """
        params:
            size (int) : size of the center crop
        """
        self.size = size
        
    def __call__(self, img, target):
        return (tF.five_crop(img, self.size)[4], 
                tF.five_crop(target, self.size)[4])
    
class JointRandomResizedCrop(object):
    def __init__(self, minimum_scale, maximum_scale, size):
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.size = size
        return
    def __call__(self, img, target):
        scale = np.random.uniform(self.minimum_scale, self.maximum_scale)
        resizedShape = (int(img.size[1] * scale), int(img.size[0] * scale))
        resizedImg = tF.resize(img, resizedShape)
        resizedTarget = tF.resize(target, resizedShape)
        
        cropSize = min(self.size, img.size[1], img.size[0])
        cropX = random.randint(0, img.size[1] - cropSize + 1)
        cropy = random.randint(0, img.size[0] - cropSize + 1)
        
        resizeImg = tF.resized_crop(resizedImg, cropX, cropy, cropSize, cropSize, self.size)
        resizeTarget = tF.resized_crop(resizedTarget, cropX, cropy, cropSize, cropSize, self.size)
        
        return resizeImg, resizeTarget  

norm = ([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]) 
class JointNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, img, target):
        return tF.normalize(img, self.mean, self.std), target