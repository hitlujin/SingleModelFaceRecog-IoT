import pymysql
from timeit import default_timer
from DBUtils.PersistentDB import PersistentDB
import datetime

Pool = PersistentDB(

)


class FaceSql(object):
    def __init__(self,log_time=True,log_label='all time'):
        self.log_time = log_time
        self.log_label = log_label


    def check_it(self):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False

        start = default_timer()
        cursor.execute("select count(id) as total from t_face_expression_recognition")
        data = cursor.fetchone()
        conn.close()
        print("-- 当前数量: %d " % data['total'])
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))

    # def inser_face_filename(self,filename,expression_code,match_rate,detection_rate,recognition_rate,video_id=1):
    #     conn = Pool.connection()
    #     cursor = conn.cursor()
    #     conn.autocommit = False

    #     conn = conn
    #     cursor = cursor
    #     start = default_timer()
    #     sqlstr = "insert into t_face_expression_recognition(file_name, expression_code,match_rate,face_detection_rate,face_recognition_rate,video_id) values(%s, %s, %s, %s, %s, %s)"
    #     params = (filename, expression_code, match_rate, detection_rate, recognition_rate, video_id)
    #     cursor.execute(sqlstr, params)
    #     conn.commit()
    #     conn.close()
    #     end = default_timer()
    #     if self.log_time:
    #         print('-- %s: %.6f 秒' % (self.log_label, end - start))

    def inser_face_filename(self,video_id,photo_id,photo_name,file_name,expression_code,match_rate=0.9,
                            face_recognition_rate=0.9,expression_recognition_rate=0.9):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False
        conn = conn
        cursor = cursor
        start = default_timer()
        sqlstr = "insert into t_face_expression_recognition(video_id,photo_id,photo_name,file_name,expression_code,match_rate,face_recognition_rate,expression_recognition_rate) values(%s, %s, %s, %s, %s, %s,%s,%s)"
        params = (video_id,photo_id,photo_name,file_name,expression_code,match_rate,face_recognition_rate,expression_recognition_rate)
        cursor.execute(sqlstr, params)
        conn.commit()
        conn.close()
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))

    def inser_face_filename_fromvideo(self,source,photo_id,photo_name,file_name,expression_code,match_rate=0.9,
                            face_recognition_rate=0.9,expression_recognition_rate=0.9):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False
        conn = conn
        cursor = cursor
        start = default_timer()
        sqlstr = "insert into t_face_expression_recognition(source,photo_id,photo_name,file_name,expression_code,match_rate,face_recognition_rate,expression_recognition_rate) values(%s, %s, %s, %s, %s, %s,%s,%s)"
        params = (source,photo_id,photo_name,file_name,expression_code,match_rate,face_recognition_rate,expression_recognition_rate)
        print("insert in mysql ")
        cursor.execute(sqlstr, params)
        conn.commit()
        conn.close()
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))

    def inser_face_feature(self,name,photo_file_name,facial_feature,describe):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False

        conn = conn
        cursor = cursor
        start = default_timer()
        sqlstr = "insert into t_photo(photo_name,photo_url, photo_file_name,facial_feature,photo_desc) values(%s, %s, %s,%s,%s)"
        params = (name, photo_file_name,photo_file_name, facial_feature,describe)
        cursor.execute(sqlstr, params)
        conn.commit()
        conn.close()
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))

    def get_all_feature(self):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False

        start = default_timer()
        cursor.execute("select photo_id,photo_name,facial_feature,photo_url from t_photo")
        data = cursor.fetchall()
        conn.close()
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))
        return data

    def inser_video_record(self,saveFileName,uploadFileName, uploadTime,uploadUserId,status):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False

        conn = conn
        cursor = cursor
        start = default_timer()
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        sqlstr = "insert into t_upload_video(save_file_name,upload_file_name, upload_time,upload_user_id,status) values(%s, %s, %s,%s,%s)"
        params = (saveFileName,uploadFileName, now,uploadUserId,status)
        cursor.execute(sqlstr, params)
        conn.commit()
        conn.close()
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))

    def update_video_record(self,save_file_name):
        conn = Pool.connection()
        cursor = conn.cursor()
        conn.autocommit = False

        conn = conn
        cursor = cursor
        start = default_timer()
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        sqlstr = "update t_upload_video set status=%s,end_time=%s where save_file_name=%s"
        params = (3,now,save_file_name)
        cursor.execute(sqlstr, params)
        conn.commit()
        conn.close()
        end = default_timer()
        if self.log_time:
            print('-- %s: %.6f 秒' % (self.log_label, end - start))
            "UPDATE table_name SET field1=%s, ..., field10=%s WHERE id=%s"