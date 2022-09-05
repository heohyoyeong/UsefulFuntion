import utils.ImgFunction as imgs
import utils.PathFunction as paths

# base_path = "D:/work_Report/report/STMResearch/MultiReference/"
# targets = ["que","ref0","ref2","ref3","ref4","ref5","ref6","ref7"]
# result = []
#
# for target in targets:
#     origin_path = base_path + f"/dataset/{target}/crop_img/*"
#     save_path = base_path + f"/renamedataset/{target}"
#     paths.change_name(origin_path, save_path)
#     result = imgs.FindWrongImg(save_path+"/*",result,40,0.01)
#
# result = set(result) #집합set으로 변환
# result = list(result)
#
# for target in targets:
#     path = base_path + f"/renamedataset/{target}/"
#     paths.remove_file(path,result)
#
# base_path = "D:/work_Report/report/STMResearch/MultiReference/STMResult/"
# save_path = "D:/work_Report/report/STMResearch/MultiReference/STMResultFusion/"
# targets = ["ref5","ref6","ref3","ref7","ref4"]
#

# ref1_name = imgs.PathToDataName(base_path + f"STMResult/ref1/*")

# fusion = imgs.MasklistDotProduct(ref1,ref2)
# ref1 = imgs.PathToImgList(base_path + "baseline/*")
# ref1_name = imgs.PathToDataName(base_path + "baseline/*")
#
# for idx in range(len(targets)):
#     plus = targets[idx]
#     ref2 = imgs.PathToImgList(base_path + f"{plus}/*")
#     fusion = imgs.MasklistDotProduct(ref1,ref2)
#     for name in range(len(ref1_name)):
#         cv2.imwrite(save_path+str(idx+1)+"/"+ref1_name[name],fusion[name])
#     ref1 = fusion

# targets = ["ref1","ref2","ref5","ref6","ref3","ref7","ref4"]
# base_path = "D:/work_Report/report/STMResearch/MultiReference/STMScoreMap/"
# save_base_path = "D:/work_Report/report/STMResearch/MultiReference/STMMeanThresScoreMap/"
# ref_name = imgs.PathToDataName(base_path + "ref1/*")
# base_thress = [z/10 for z in range(25,35)]
#
# for th in base_thress:
#     for case in range(2,len(targets)+1):
#         datas = []
#         for idx in range(case):
#             img_list = imgs.PathToImgList(base_path + targets[idx]+"/*")
#             datas.append(img_list)
#         mean = imgs.ImgAxisMean(datas)
#         # min = imgs.ImgAxisMin(datas)
#         threshold = 36.4*th
#         thres_img = imgs.ImgThreshold(mean,threshold)
#         save_path = save_base_path + str(th)+"/"+str(case)
#         imgs.NamePathImgSave(thres_img,save_path,ref_name)
#
#     for case in range(2,len(targets)+1):
#         GT_Path = save_base_path + "gt/*"
#         Mask_Path = save_base_path + str(th)+"/"+str(case) +"/*"
#         result = imgs.CalcScore(GT_Path, Mask_Path)
#         TruePositiveRate,FalseNegativeRate = imgs.CalcFinalScore(result)
#
#         # print(f"{case}의 마스크를 조합하여 나온 결과")
#         # print(f"검출률 :{TruePositiveRate}, 과검률 : {FalseNegativeRate}")
#         print(f"{case},{TruePositiveRate},{FalseNegativeRate}")
#     print("")

path = r"D:/work_Report/report/STMResearch/FilterMapPreProcess/0829 dataset/train_dataset/1/mask/"

imgs.PathImgResize(path+"*",path,224,224)