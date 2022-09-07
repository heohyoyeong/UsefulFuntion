from glob import glob
import cv2
import matplotlib.pyplot as plt
import os


def change_name(origin_path,save_path):
    """ 원본 파일의 index 순으로 이름을 변경해서 저장해주는 기능 """
    img_list = glob(origin_path)
    os.makedirs(save_path,exist_ok=True)
    for idx in range(len(img_list)):
        img = cv2.imread(img_list[idx], cv2.IMREAD_COLOR)
        origin_name = img_list[idx].split("\\")[-1]
        origin_type = origin_name.split(".")[-1]
        name = str(idx).zfill(5) + "." + origin_type
        cv2.imwrite(save_path+"/"+name,img)


def Calc_diff_MaxScore(Apath,Bpath,save_path):
    """ 이름끝에 붙은 ScoreMap max값을 뺴서 CSV 파일로 저장 """
    A_list = glob(Apath)
    B_list = glob(Bpath)

    scores = []
    names = []
    for idx in range(len(A_list)):
        A_name = (A_list[idx].split("\\")[-1]).split(".jpg")[0]
        B_name = (B_list[idx].split("\\")[-1]).split(".jpg")[0]

        A_Score = float(A_name.split("_")[-1])
        B_Score = float(B_name.split("_")[-1])

        name = A_name.split("_")[0] + "_" +A_name.split("_")[1]
        diff = A_Score-B_Score
        scores.append(diff)
        names.append(name)
        diff = round(diff,4)

        os.makedirs(save_path+"bigDiff", exist_ok=True)
        os.makedirs(save_path+"bigDiff/rect", exist_ok=True)
        os.makedirs(save_path+"bigDiff/square", exist_ok=True)

        if (diff<=-0.25 or diff>=0.25):
            rect_list = glob(save_path + "rect/*/" + name + "*.jpg")
            square_list = glob(save_path + "square/*/" + name + "*.jpg")
            for i in range(len(rect_list)):
                fusion_rect = cv2.imread(rect_list[0])
                resize_rect = cv2.imread(rect_list[1])
                origin_rect = cv2.imread(rect_list[2])
                socremap_rect = cv2.imread(rect_list[3])
                maxscore_rect = rect_list[3].split("\\")[-1].split(".jpg")[0].split("_")[-1]
                os.makedirs(save_path + "bigDiff/rect/"+name, exist_ok=True)
                cv2.imwrite(save_path + "bigDiff/rect/"+name+"/fusion_rect.jpg",fusion_rect)
                cv2.imwrite(save_path + "bigDiff/rect/"+name+"/resize_rect.jpg",resize_rect)
                cv2.imwrite(save_path + "bigDiff/rect/"+name+"/origin_rect.jpg",origin_rect)
                cv2.imwrite(save_path + "bigDiff/rect/"+name+"/"+maxscore_rect+"_rect.jpg",socremap_rect)

                fusion_square = cv2.imread(square_list[0])
                resize_square = cv2.imread(square_list[1])
                origin_square = cv2.imread(square_list[2])
                socremap_square = cv2.imread(square_list[3])
                maxscore_square = square_list[3].split("\\")[-1].split(".jpg")[0].split("_")[-1]
                os.makedirs(save_path + "bigDiff/square/"+name, exist_ok=True)
                cv2.imwrite(save_path + "bigDiff/square/"+name+"/fusion_square.jpg",fusion_square)
                cv2.imwrite(save_path + "bigDiff/square/"+name+"/resize_square.jpg",resize_square)
                cv2.imwrite(save_path + "bigDiff/square/"+name+"/origin_square.jpg",origin_square)
                cv2.imwrite(save_path + "bigDiff/square/"+name+"/"+maxscore_square+"_square.jpg",socremap_square)

                fig = plt.figure(figsize=(10, 10))
                rows = 2
                cols = 4

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.set_title("Rect Origin")
                image1 = ax1.imshow(origin_rect, cmap='gray')

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.set_title("Rect Resize")
                image2 = ax2.imshow(resize_rect, cmap='gray')

                ax3 = fig.add_subplot(rows, cols, 3)
                ax3.set_title("Rect ScoreMap")
                image3 = ax3.imshow(socremap_rect, cmap='gray')

                ax4 = fig.add_subplot(rows, cols, 4)
                ax4.set_title("Rect ScoreMap Fusion")
                image4 = ax4.imshow(fusion_rect[:, :, ::-1], cmap='coolwarm')


                ax5 = fig.add_subplot(rows, cols, 5)
                ax5.set_title("Square Origin")
                image5 = ax5.imshow(origin_square, cmap='gray')

                ax6 = fig.add_subplot(rows, cols, 6)
                ax6.set_title("Square Resize")
                image6 = ax6.imshow(resize_square, cmap='gray')

                ax7 = fig.add_subplot(rows, cols, 7)
                ax7.set_title("Square ScoreMap")
                image7 = ax7.imshow(socremap_square, cmap='gray')

                ax8 = fig.add_subplot(rows, cols, 8)
                ax8.set_title("Square ScoreMap Fusion")
                image8 = ax8.imshow(fusion_square[:, :, ::-1], cmap='coolwarm')

                fig.tight_layout()
                if (diff>0):
                    fig.savefig(save_path + "bigDiff/img/square/" + name + "_" + str(diff) + ".jpg")
                else:
                    fig.savefig(save_path + "bigDiff/img/rect/" + name + "_" + str(diff) + ".jpg")
                plt.close(fig)

def remove_file(path, name_list):
    for name in name_list:
        os.remove(path+name)