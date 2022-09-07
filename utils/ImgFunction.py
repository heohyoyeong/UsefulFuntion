import cv2
from glob import glob
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def Template_Maching(Small_Image,Large_Image):
    """ 큰이미지와 작은 이미지를 받아 Template Matching하여 좌상단 좌표 출력 """
    res = cv2.matchTemplate(Large_Image, Small_Image, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    return [top_left[0], top_left[1]]

def ImgCrop(Image,x,y,w,h):
    """ 이미지와 ROI를 Crop 이미지 반환"""
    CropImage = Image[y:y+h,x:x+w]
    return CropImage

def ImgDotProduct(Img1,Img2):
    """ Img1과 Img2를 DotProduct하여 반환"""
    img = Img1*Img2
    return img

def ImgAxisMean(ImgList):
    ImgArray = np.array(ImgList)
    MeanImg = np.mean(ImgArray,axis=0)
    return MeanImg

def ImgAxisMin(ImgList):
    ImgArray = np.array(ImgList)
    MinImg = np.min(ImgArray,axis=0)
    return MinImg

def MaskMasker(SavePath, ROI, w, y):
    """ ROI와 Size를 받아 mask를 저장"""
    mask = np.zeros((y,w))
    for idx in range(len(ROI)):
        mask[ROI[idx][1] : ROI[idx][1] + ROI[idx][3], ROI[idx][0]:ROI[idx][0] + ROI[idx][2]] = 255
    cv2.imwrite(SavePath,mask)

def CalcSimilarity(Img1,Img2):
    """ Img1과 Img2를 받아 유사도"""
    (score, _) = ssim(Img1, Img2, full=True)
    return score

def PathToImgList(Img_Path):
    """ Path를 받아 grayscale img list 반환"""
    result = []
    img_Path_list = glob(Img_Path)
    for path in img_Path_list:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        result.append(img)
    return result

def PathToDataName(Data_Path):
    """ Path를 받아 내부 파일 이름 list 반환"""
    result = []
    Path_list = glob(Data_Path)
    for path in Path_list:
        name = path.split("\\")[-1]
        result.append(name)
    return result

def NamePathImgSave(ImgList,SavePath,DataName):
    """ 이미지와 저장주소 데이터 명을 받아 저장"""
    os.makedirs(SavePath, exist_ok=True)
    for i in range(len(ImgList)): cv2.imwrite(SavePath+"/"+DataName[i],ImgList[i])

def ImgThreshold(ImgList,Threshold):
    result = []
    for img in ImgList:
        _, dst = cv2.threshold(img, Threshold, 255, cv2.THRESH_BINARY)
        result.append(dst)
    return result

################################################################
def save_diff_img(que_path,ref_path,save_path):
    """ 2개의 이미지 주소를 받아 차영상후 idx 순으로 저장 """

    que_img_list = PathToImgList(que_path)
    ref_img_list = PathToImgList(ref_path)
    que_name_list = PathToDataName(que_path)

    for idx in range(len(que_name_list)):
        name = que_name_list[idx]
        que = que_img_list[idx]
        ref = ref_img_list[idx]
        diff = cv2.absdiff(que, ref)
        cv2.imwrite(save_path+name, diff)

def SaveMaskImgFusion(img_path,mask_path,save_path):
    """ 이미지와 Mask를 받아 퓨전하여 img위에 Jet이미지를 붙여 넣어줌 """
    
    que_img_list = PathToImgList(img_path)
    mask_img_list = PathToImgList(mask_path)
    que_name_list = PathToDataName(img_path)

    for idx in range(len(que_name_list)):
        name = que_name_list[idx]
        img = que_img_list[idx]
        mask = mask_img_list[idx]

        img = np.float32(img)/255
        heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)/255
        cam = heatmap + np.float32(img)
        cam = cam/np.max(cam)
        cv2.imwrite(save_path+name, np.uint8(255*cam))

def PathImgResize(InputPath,SavePath,w,h):
    """ 주소를 받아 이미지를 Resize한 후 저장할 위치에 저장"""
    que_img_list = PathToImgList(InputPath)
    que_name_list = PathToDataName(InputPath)

    for idx in range(len(que_name_list)):
        name = que_name_list[idx]
        img = que_img_list[idx]
        ResizeImg = cv2.resize(img,(w,h))
        cv2.imwrite(SavePath+name,ResizeImg)

def Roi_Calc(ROI,Margin_Size,Image):
    """ 이미지와 ROI를 받아 Margin사이즈를 추가한 ROI를 계산 (이미지 최대 크기 보정 기능 추가)"""

    w, h = Image.shape

    margined_x1 = ROI[1] - Margin_Size
    margined_y1 = ROI[0] - Margin_Size
    margined_x2 = ROI[1] + ROI[3] + Margin_Size
    margined_y2 = ROI[0] + ROI[2] + Margin_Size

    if (margined_y1 < 0): margined_y1 = 0
    if (margined_x1 < 0): margined_x1 = 0
    if (margined_y2 > h): margined_y1 = h
    if (margined_x2 > w): margined_x2 = w

    return [margined_y1,margined_x1,margined_y2,margined_x2]

def InputImageTemplateMachingProcess(Query_Origin,Reference1_Origin,Reference2_Origin, MaskOrigin, ROI,Margin_Size):
    """ que이미지를 입력 ROI를 Crop한 후
    Margin Size를 추가한 크기의 레퍼런스에서 찾아
    TemplateMatching을 수행후 3개의 이미지를 리턴 """

    Query_Crop = Query_Origin[ROI[1]:ROI[1]+ROI[3],ROI[0]:ROI[0]+ROI[2]]
    Mask_Crop = MaskOrigin[ROI[1]:ROI[1]+ROI[3],ROI[0]:ROI[0]+ROI[2]]

    Ref1_Semi = Roi_Calc(ROI, Margin_Size, Reference1_Origin)
    Ref1_Semi_Crop = Reference1_Origin[Ref1_Semi[1]:Ref1_Semi[3],Ref1_Semi[0]:Ref1_Semi[2]]
    Ref1_ROI = Template_Maching(Query_Crop,Ref1_Semi_Crop)
    Ref1_Crop = Ref1_Semi_Crop[Ref1_ROI[1]:Ref1_ROI[1] + ROI[3], Ref1_ROI[0]:Ref1_ROI[0] + ROI[2]]

    Ref2_Semi = Roi_Calc(ROI, Margin_Size, Reference2_Origin)
    Ref2_Semi_Crop = Reference2_Origin[Ref2_Semi[1]:Ref2_Semi[3],Ref2_Semi[0]:Ref2_Semi[2]]
    Ref2_ROI = Template_Maching(Query_Crop,Ref2_Semi_Crop)
    Ref2_Crop = Ref2_Semi_Crop[Ref2_ROI[1]:Ref2_ROI[1] + ROI[3], Ref2_ROI[0]:Ref2_ROI[0] +ROI[2]]

    return Query_Crop, Ref1_Crop, Ref2_Crop, Mask_Crop

def CalcScoreMapMeanDiff(ScoreMapPath,MaskPath):
    '''검출된 결함의 ScoreMap 평균과 그외 지역의 평균의 차를 계산'''
    result = []
    ScoreMaplist = PathToImgList(ScoreMapPath)
    Masklist = PathToImgList(MaskPath)

    for idx in range(len(Masklist)):
        ScoreMap = ScoreMaplist[idx]
        Mask = Masklist[idx]
        if(len(np.nonzero(Mask)[0])==0):
            result.append(0)
        else:
            BackgroundMean = np.mean(ScoreMap[np.nonzero(255-Mask)])
            DetectionMean = np.mean(ScoreMap[np.nonzero(Mask)])
            result.append(DetectionMean-BackgroundMean)
    return result

def CalcIOU(GroundTruthPath,MaskPath):
    '''GroundTruthPath와 MaskPath를 입력받아 해당 데이터의 결함의 수와 검출된 결함의 IOU 출력'''
    result = []
    GroundTruthlist = PathToImgList(GroundTruthPath)
    Masklist = PathToImgList(MaskPath)
    for idx in range(len(Masklist)):
        Base = []
        OriginGroundTruth = GroundTruthlist[idx]
        OriginMask = Masklist[idx]
        _, _, GroundTruthstats, _ = cv2.connectedComponentsWithStats(OriginGroundTruth)
        _, _, Maskstats, _ = cv2.connectedComponentsWithStats(OriginMask)
        RectGroundTruth = np.zeros(OriginMask.shape,np.uint8)
        RectMask = np.zeros(OriginMask.shape,np.uint8)
        Base.append(len(GroundTruthstats)-1)
        for i in range(1,len(Maskstats)):
            x = Maskstats[i][0]
            y = Maskstats[i][1]
            w = Maskstats[i][2]
            h = Maskstats[i][3]
            cv2.rectangle(RectMask, (x, y, w, h), (120),-1)

        for i in range(1,len(GroundTruthstats)):
            x = GroundTruthstats[i][0]
            y = GroundTruthstats[i][1]
            w = GroundTruthstats[i][2]
            h = GroundTruthstats[i][3]
            cv2.rectangle(RectGroundTruth, (x, y, w, h), (135),-1)

        FusionMask = (RectGroundTruth+RectMask)
        _, _, Fusionstats, _ = cv2.connectedComponentsWithStats(FusionMask)
        for i in range(1,len(Fusionstats)):
            x = Fusionstats[i][0]
            y = Fusionstats[i][1]
            w = Fusionstats[i][2]
            h = Fusionstats[i][3]
            a = Fusionstats[i][4]
            FusionCrop = ImgCrop(FusionMask,x,y,w,h)
            FusionCrop[FusionCrop!=255]=0
            oh = len(FusionCrop[FusionCrop==255])
            if oh>1:
                Base.append([x, y, w, h, oh / a])
        result.append(Base)
    return result

def CalcScore(GroundTruthPath,MaskPath):
    '''GroundTruthPath와 MaskPath를 입력받아 검출, 미검, 과검 수 출력'''
    result = []
    GroundTruthlist = PathToImgList(GroundTruthPath)
    Masklist = PathToImgList(MaskPath)
    for idx in range(len(Masklist)):
        TP = 0
        FN = 0
        FP = 0

        OriginGroundTruth = GroundTruthlist[idx]
        OriginMask = Masklist[idx]
        _, _, GroundTruthstats, _ = cv2.connectedComponentsWithStats(OriginGroundTruth)
        kernel = np.ones((7, 7), np.uint8)
        BlurGroundTruth = cv2.dilate(OriginGroundTruth, kernel)
        BlurGroundTruth = cv2.morphologyEx(BlurGroundTruth, cv2.MORPH_CLOSE, kernel)
        BlurGroundTruth[np.nonzero(BlurGroundTruth)]=155
        OriginMask[np.nonzero(OriginMask)]=100

        FusionMask = (BlurGroundTruth+OriginMask)
        _, _, Fusionstats, _ = cv2.connectedComponentsWithStats(FusionMask)
        for i in range(1,len(Fusionstats)):
            x = Fusionstats[i][0]
            y = Fusionstats[i][1]
            w = Fusionstats[i][2]
            h = Fusionstats[i][3]
            a = Fusionstats[i][4]
            FusionCrop = ImgCrop(FusionMask,x,y,w,h)
            if (len(FusionCrop[FusionCrop==255])>0):
                TP +=1
            elif (np.mean(FusionCrop[FusionCrop!=0])>130):
                FN +=1
            else:
                FP +=1
        if (TP+FN!=len(GroundTruthstats)-1):
            TP = len(GroundTruthstats)-1 - FN
        result.append([TP,FN,FP])
    return result

def CalcFinalScore(DetectionResult):
    '''CalcScore의 결과를 받아 해당 데이터셋의 검출률과 과검률 출력'''
    AddResult = np.sum(DetectionResult,axis=0)
    TruePositiveRate = AddResult[0]/(AddResult[1]+AddResult[0]) * 100
    FalseNegativeRate = AddResult[2]/AddResult[0] * 100


    return TruePositiveRate,FalseNegativeRate