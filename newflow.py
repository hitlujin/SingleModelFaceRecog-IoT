import insightface
import cv2
import tensorflow as tf
from skimage import transform
from facesql import FaceSql
from sklearn.preprocessing import normalize
import numpy as np
from flask import Flask,request
import time
from classify import Classify
from timeit import default_timer
import json
from multiprocessing import Value
from ctypes import c_wchar_p 
import threading
import datetime

import os
import shutil



det = insightface.model_zoo.SCRFD("det.onnx")
face_model = insightface.model_zoo.get_model("facerec.onnx")
mysql = FaceSql()
classify = Classify("best_model.hdf5")

video0_last = Value('l',0)
video1_last = Value('l',0)
video0_counter = Value('i',0)  
video1_counter = Value('i',0)
video0_result = Value(c_wchar_p,"")
video1_result = Value(c_wchar_p,"")

expression_title = {
    0: "生气",
    1: "沮丧",
    2: "恐惧",
    3: "快乐",
    4: "无表情",
    5: "伤心",
    6: "惊讶"
}

def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
    ret = []
    for landmark in landmarks:
        # landmark = np.array(landmark).reshape(2, 5)[::-1].T
        tform.estimate(landmark, src)
        ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
    ret = np.transpose(ret, axes=[0,3,1,2])
    return (np.array(ret) * 255).astype(np.uint)

def do_detect_in_image(image, det, image_format="BGR"):
    imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
    imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
    bboxes, pps = det.detect(imm_BGR, (640, 640))
    nimgs = face_align_landmarks_sk(imm_RGB, pps)
    bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
    return bbs, ccs,pps, nimgs

def handle_img(img,source):
    bbs,ccs,pps,imgs = do_detect_in_image(img,det)
    nimgs = imgs.astype(np.float32)
    embeddings = normalize(face_model.forward(nimgs))

    # read sql for face embedding
    names = []
    embeddings_sql = []
    person_id = []
    id_files = []
    features = mysql.get_all_feature()
    for i,name,raw_feature,file_name in features:
        feature = np.frombuffer(raw_feature,dtype=np.float32)
        names.append(name)
        embeddings_sql.append(feature)
        person_id.append(i)
        id_files.append(file_name)
    embeddings_sql = np.array(embeddings_sql)


    score = np.dot(embeddings,embeddings_sql.T)
    index = np.argmax(score,axis=1)
    return_names = []
    return_ids = []
    return_filenames = []
    sql_id_for_insert = []
    sql_name_for_insert = []
    i = 0
    for k in index:
        if score[i][k] > 0.5:
            return_filenames.append(id_files[k])
            return_names.append(names[k])
            sql_name_for_insert.append(names[k])
            return_ids.append(person_id[k])
            sql_id_for_insert.append(person_id[k])
        else:
            return_filenames.append("")
            return_names.append("")
            sql_name_for_insert.append(None)
            return_ids.append("")
            sql_id_for_insert.append(None)
        i += 1
    nimgs = np.transpose(nimgs, axes=[0,2,3,1])
    class_result = classify.classify_batch(nimgs)
    express_index = np.argmax(class_result,axis=1)
    expression_name = []
    for i in express_index:
        expression_name.append(expression_title[i])
    int_pps = []
    for pp in pps:
        int_pp = []
        for point in pp:
            int_point = []
            for item in point:
                int_point.append(int(item))
            int_pp.append(int_point)
        int_pps.append(int_pp)
    # print(int_pps)

    int_bbs = []
    for bb in bbs:
        int_bb = []
        for item in bb:
            int_bb.append(int(item))
        int_bbs.append(int_bb)
    file_names = []
    nn = 0
    print(len(nimgs))
    for i,img in enumerate(nimgs):
        file_name = str(int(time.time() * 1000)) + str(i) + '.jpg'
        save_path = '/usr/local/aiivr/exppic/' + file_name
        file_names.append(file_name)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        nn += 1
        cv2.imwrite(save_path,img)
        print(source,sql_id_for_insert[i],sql_name_for_insert[i],file_name,express_index[i],ccs[i],score[i][index[i]],float(class_result[i][express_index[i]]))
        mysql.inser_face_filename_fromvideo(source=source,photo_id=sql_id_for_insert[i],photo_name=sql_name_for_insert[i],
                            file_name=file_name,expression_code=express_index[i],match_rate=ccs[i],
                            face_recognition_rate=score[i][index[i]],expression_recognition_rate=float(class_result[i][express_index[i]]))
    print(nn)
    return None

def print_image(frame,saveImagePath):
    #detect face and save path
    bbs,ccs,pps,imgs = do_detect_in_image(img,det)
    nimgs = imgs.astype(np.float32)
    nimgs = np.transpose(nimgs, axes=[0,2,3,1])
    class_result = classify.classify_batch(nimgs)
    express_index = np.argmax(class_result,axis=1)
    expression_name = []
    for i in express_index:
        expression_name.append(expression_title[i])
    




def handle_video(saveFileName):
    video = cv2.VideoCapture()
    file_name = '/usr/local/aiivr/exppic/videofile/' + saveFileName
    video.open(file_name)
    count = 0
    # rm tmp dir and recreate it
    if os.path.exists('/usr/local/aiivr/exppic/tmp'):
        shutil.rmtree('/usr/local/aiivr/exppic/tmp')
    os.mkdir('/usr/local/aiivr/exppic/tmp')

    while True:
        _, frame = video.read()
        count += 1
        if frame is None:
            break
        if count % 30 == 0:
            saveImagePath = '/usr/local/aiivr/exppic/tmp/' + str(count) + '.jpg'
            handle_img(frame,source=saveFileName)
            print_image(frame,saveImageName)
    # determin file name 
    # merge all image in tmp folder into a video using ffmpeg
    os.system('ffmpeg -r 30 -i /usr/local/aiivr/exppic/tmp/%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p /usr/local/aiivr/exppic/videofile/' + saveFileName + '.mp4')
    video.release()
    mysql.update_video_record(saveFileName)
    return None

app = Flask(__name__)
@app.route('/api/regitface',methods=['POST'])
def regitface():
    data = json.loads(request.data)
    name = data["photoName"]
    describe = ""
    if data["photoDesc"] is not None:
        describe = data["photoDesc"]
    # raw_img = data["imageBase64String"]
    file_name = data["photoFileName"]
    save_path = "/usr/local/aiivr/photo/" + file_name
    img = cv2.imread(save_path)
    # nprr = np.fromstring(raw_img,np.uint8)
    # img = cv2.imdecode(nprr,cv2.IMREAD_COLOR)
    bbs,ccs,pps,imgs = do_detect_in_image(img,det)
    # print("det time: ",det_end - det_start)
    nimgs = imgs.astype(np.float32)
    a1 = normalize(face_model.forward(nimgs))

    data = a1[0].tobytes()

    # save photo
    file_name = str(int(time.time() * 1000)) + '.jpg'
    save_path = '/usr/local/aiivr/photo/' + file_name
    imgs = np.transpose(nimgs, axes=[0,2,3,1])
    img = cv2.cvtColor(imgs[0],cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path,img)
    mysql.inser_face_feature(name,file_name,data,describe)

    response_dict = {
        'code':200,
        'msg':"sucess",
        'data':""
    }
    return response_dict

@app.route('/api/videofeatureextract',methods=['POST'])
def vidieo():
    # post param
    data = json.loads(request.data)
    saveFileName = data['saveFileName']
    uploadFileName = data['uploadFileName']
    uploadTime = data['uploadTime']
    uploadUserId = data['uploadUserId']
    status = data['status']
    print(saveFileName,uploadFileName,uploadTime,uploadUserId)
    mysql.inser_video_record(saveFileName,uploadFileName, uploadTime,uploadUserId,status=2)
    # handle_video(saveFileName)
    t = threading.Thread(target=handle_video,args=(saveFileName,))
    t.start()
    response_dict = {
        'code':200,
        'msg':"sucess",
        'data':""
    }
    return response_dict

@app.route('/api/facefeatureextract',methods=['POST'])
def demo():
    start_time = time.time()
    r = request

    # sample frame
    videoId = int(r.args['videoId'])
    predict_flag = False
    sample_frequent = 15
    if videoId == 1:
        with video0_counter.get_lock():
            if video0_counter.value % sample_frequent == 0:
                predict_flag = True
                # print(video0_counter.value)
            video0_counter.value += 1
    else:
        with video1_counter.get_lock():
            if video1_counter.value % sample_frequent == 0:
                predict_flag = True
            video1_counter.value += 1

    if predict_flag:  
        nprr = np.fromstring(r.data,np.uint8)
        img = cv2.imdecode(nprr,cv2.IMREAD_COLOR)
        det_start = default_timer()
        bbs,ccs,pps,imgs = do_detect_in_image(img,det)
        det_end = default_timer()
        # print("det time:",det_end - det_start)
        nimgs = imgs.astype(np.float32)
        rec_start = default_timer()
        embeddings = normalize(face_model.forward(nimgs))
        rec_end = default_timer()
        # print("rec time:",rec_end - rec_start)
        names = []
        embeddings_sql = []
        person_id = []
        id_files = []
        features = mysql.get_all_feature()
        for i,name,raw_feature,file_name in features:
            feature = np.frombuffer(raw_feature,dtype=np.float32)
            names.append(name)
            embeddings_sql.append(feature)
            person_id.append(i)
            id_files.append(file_name)
        embeddings_sql = np.array(embeddings_sql)
        score = np.dot(embeddings,embeddings_sql.T)
        index = np.argmax(score,axis=1)


        return_names = []
        return_ids = []
        return_filenames = []
        sql_id_for_insert = []
        sql_name_for_insert = []
        i = 0
        for k in index:
            if score[i][k] > 0.5:
                return_filenames.append(id_files[k])
                return_names.append(names[k])
                sql_name_for_insert.append(names[k])
                return_ids.append(person_id[k])
                sql_id_for_insert.append(person_id[k])
            else:
                return_filenames.append("")
                return_names.append("")
                sql_name_for_insert.append(None)
                return_ids.append("")
                sql_id_for_insert.append(0)
            i += 1
        nimgs = np.transpose(nimgs, axes=[0,2,3,1])
        cla_start = default_timer()
        class_result = classify.classify_batch(nimgs)
        cla_end = default_timer()
        # print('class time:',cla_end - cla_start)
        express_index = np.argmax(class_result,axis=1)
        expression_name = []
        for i in express_index:
            expression_name.append(expression_title[i])

        int_pps = []
        for pp in pps:
            int_pp = []
            for point in pp:
                int_point = []
                for item in point:
                    int_point.append(int(item))
                int_pp.append(int_point)
            int_pps.append(int_pp)
        # print(int_pps)

        int_bbs = []
        for bb in bbs:
            int_bb = []
            for item in bb:
                int_bb.append(int(item))
            int_bbs.append(int_bb)
        # print(int_bbs)


        if videoId == 1:
            if int(time.time() * 1000) - video0_last.value > 10000:
                # print(video0_last.value)
                save_flag = True
                # print(int(time.time() * 1000) - video0_last.value )
                with video0_last.get_lock():
                    video0_last.value = int(time.time() * 1000)
            else:
                save_flag = False
        else:
            if int(time.time() * 1000) - video1_last.value > 10000:
                save_flag = True
                # print(int(time.time() * 1000) - video1_last.value)
                with video1_last.get_lock():
                    video1_last.value = int(time.time() * 1000)
            else:
                save_flag = False

        file_names = []
        if save_flag:
            for i,img in enumerate(nimgs):
                file_name = str(int(time.time() * 1000)) + str(i) + '.jpg'
                save_path = '/usr/local/aiivr/exppic/' + file_name
                file_names.append(file_name)
                # print(save_path)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path,img)
                # print(videoId,sql_id_for_insert[i],sql_name_for_insert[i],file_name,express_index[i],ccs[i],score[i][index[i]],float(class_result[i][express_index[i]]))
                mysql.inser_face_filename(video_id=videoId,photo_id=sql_id_for_insert[i],photo_name=sql_name_for_insert[i],
                                    file_name=file_name,expression_code=express_index[i],match_rate=ccs[i],
                                    face_recognition_rate=score[i][index[i]],expression_recognition_rate=float(class_result[i][express_index[i]]))

        result = []
        for i in range(len(bbs)):
            item = {}
            item["bbox"] = int_bbs[i]
            item['face_detect_score'] = ""
            item['landmark'] = int_pps[i]
            item['expression'] = expression_name[i] + "  " + return_names[i]
            result.append(item)

        expressions = []
        for i in range(len(bbs)):
            item = {}
            item["expression_code"] = int(express_index[i])
            item["name"] = return_names[i]
            # result[i]["expression"] = result[i]["expression"] + " " + return_names[i]
            item["id"] = sql_id_for_insert[i]
            if item["id"] != 0:
                item["photo_file_name"] = return_filenames[i]
            else:
                item["photo_file_name"] = "default.jpeg"
            item["registered"] = ""
            # if save_flag:
            item["filename"] = file_names[i]
            # else:
                # item["filename"] = ""
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S")
            item["collectTime"] = now
            item["matchRate"] = float(ccs[i])
            item["expressionRate"] = float(class_result[i][express_index[i]])
            expressions.append(item)

        response_dict = {
            'code':200,
            'msg':"sucess",
            'data':""
        }
        end_time = time.time()
        print("time total: ", end_time - start_time)
        response_dict["data"] = {"tags":result,"expressions":expressions}
        if videoId == 1:
            with video0_result.get_lock():
                video0_result.value = json.dumps(response_dict)
        else:
            with video1_result.get_lock():
                video1_result.value = json.dumps(response_dict)
        if save_flag:
            for i,item in enumerate(file_names):
                response_dict["data"]["expressions"][i]['filename'] = item
        print(json.dumps(response_dict))
        return json.dumps(response_dict)

    # if not in sample, use cache
    if videoId == 1:
        print(video0_result.value)
        return video0_result.value
    else:
        print(video1_result.value)
        return video1_result.value
app.run(host='0.0.0.0',port="5000")